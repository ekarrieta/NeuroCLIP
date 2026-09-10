from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


# -------------------- utils --------------------

def _trim_to_min(
    a: np.ndarray, b: np.ndarray, axis: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    T = min(a.shape[axis], b.shape[-1])
    slicer = [slice(None)] * a.ndim
    slicer[axis] = slice(0, T)
    a = a[tuple(slicer)]
    b = b[..., :T]
    return a, b

def format_sub(x: Any) -> str:
    s = str(x)
    return s if len(s) >= 2 else f"0{s}"

def story2task(task: str) -> str:
    dic = {'lw1': '0', 'cable_spool_fort': '1', 'easy_money': '2', 'the_black_willow': '3'}
    return dic.get(task, task)

def _latency_shift(feat: np.ndarray, shift_ms: float, hz: float) -> np.ndarray:
    if shift_ms == 0:
        return feat
    frames = int(round(shift_ms / 1000.0 * hz))
    if frames > 0:
        return feat[:, frames:] if feat.shape[1] > frames else feat[:, :0]
    pad = np.zeros((feat.shape[0], -frames), dtype=feat.dtype)
    return np.concatenate([pad, feat], axis=1)


# -------------------- configs --------------------

@dataclass
class DatasetConfig:
    """Loaded from readers/{dataset_name}.yaml. Drives dataset-specific behaviour."""
    name: str
    bids_root: Path
    stim_root: Path
    meg_target_hz: float = 120.0
    meg_ref_meg: bool = False
    audio_resample_hz: float = 16000.0
    cache_subdir: str = ""
    run_column: Optional[str] = None   # column name for run info (e.g. MOUS multi-run subjects)
    meg_picks: Optional[str] = None    # Channel name regex for picking (e.g. "^M[LRZ]")
    meg_exclude: List[str] = field(default_factory=list)  # Channels to exclude after picking
    meg_layout: Optional[str] = None   # MNE layout name for sensor positions (e.g. "CTF-275")
    n_channels: int = 208              # number of MEG channels after picking
    n_subjects: int = 27               # number of subjects (for subject embedding layer)

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetConfig":
        path = Path(path).expanduser().resolve()
        with path.open(encoding="utf-8") as f:
            d = yaml.safe_load(f)

        base_dir = (path.parent / d.get("path_base", ".")).resolve()

        def resolve_config_path(value: str) -> Path:
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                return candidate
            return (base_dir / candidate).resolve()

        return cls(
            name=d["name"],
            bids_root=resolve_config_path(d["bids_root"]),
            stim_root=resolve_config_path(d["stim_root"]),
            meg_target_hz=d["meg"]["target_hz"],
            meg_ref_meg=d["meg"].get("ref_meg", False),
            audio_resample_hz=d["audio"]["resample_hz"],
            meg_picks=d["meg"].get("picks"),
            meg_exclude=d["meg"].get("exclude") or [],
            meg_layout=d["meg"].get("layout"),
            cache_subdir=d.get("cache_subdir", d["name"]),
            run_column=d.get("subjects", {}).get("run_column"),
            n_channels=d["meg"].get("n_channels", 208),
            n_subjects=d.get("subjects", {}).get("n_subjects", 27),
        )


@dataclass
class MEGConfig:
    bids_root: Path
    picks: Optional[str] = None
    target_hz: float = 120.0
    cache_dir: Path = Path("cache/meg")

@dataclass
class AudioConfig:
    stim_root: Path
    resample_hz: float = 16000.0
    cache_dir: Path = Path("cache/audio")

@dataclass
class AlignConfig:
    latency_ms: float = 0.0   # features forward relative to MEG
    trim_to_min: bool = False  # enforce equal T


# -------------------- dataset --------------------
class SegmentsDataset(Dataset):
    """
    Returns dict: {
        'meg': [C, T], 'feature': [F, T], 'words': str
        ... plus some metadata
    }
    """
    def __init__(
        self,
        segments_df: pd.DataFrame,
        split: str,               # 'train': 2, 'valid': 1, 'test': 0
        feature_type: str,
        meg_cfg: MEGConfig,
        audio_cfg: AudioConfig,
        align_cfg: Optional[AlignConfig] = None,
        dataset_cfg: Optional[DatasetConfig] = None,
    ):
        required_columns = {
            "subject", "session", "task", "audio_file", "onset", "start",
            "duration", "split", "words", "anchor_word", "block_uid",
            "block_stim_uid", "segment_uid", "stim_id", "condition",
        }
        missing = sorted(required_columns - set(segments_df.columns))
        if missing:
            raise ValueError(
                "Segments table is missing required column(s): "
                + ", ".join(missing)
            )

        if split == 'train':
            self.split = 2
        elif split == 'valid':
            self.split = 1
        elif split == 'test':
            self.split = 0
        elif split == 'trf':
            self.split = 3
        else:
            raise ValueError(f"Invalid split: {split}")
        self.df = segments_df[segments_df["split"] == self.split].reset_index(drop=True)
        self.meg_cfg = meg_cfg
        self.audio_cfg = audio_cfg
        self.feature_type = feature_type
        self.align_cfg = align_cfg or AlignConfig()
        self.dataset_cfg = dataset_cfg

        # Create cache directories
        self.meg_cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_cfg.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create recording index mapping
        self._create_recording_mappings(segments_df)

    def _create_recording_mappings(self, segments_df: pd.DataFrame):
        """Create mappings for recording_index, recordings string, and subject index."""
        is_mous = self.dataset_cfg is not None and self.dataset_cfg.name == 'schoffelen2019'

        if is_mous:
            # Schoffelen subjects are "A2002" .. "A2125" (with gaps).
            # Sort by last 4 digits numerically, then assign contiguous 0-based indices.
            unique_subjects = sorted(segments_df['subject'].unique(),
                                     key=lambda s: int(str(s)[-4:]))
            self.subject_index_map = {s: i for i, s in enumerate(unique_subjects)}
        else:
            # Gwilliams subjects are ints 1-27 → old mapping: int(subject) - 1
            self.subject_index_map = {s: int(s) - 1
                                      for s in segments_df['subject'].unique()}

        # Get unique combinations of subject, session, task from the entire dataset
        unique_recordings = segments_df[['subject', 'session', 'task']].drop_duplicates()

        # Sort to ensure consistent ordering
        unique_recordings = unique_recordings.sort_values(['subject', 'session', 'task']).reset_index(drop=True)

        self.recording_index_map = {}
        self.recordings_map = {}

        is_mous = self.dataset_cfg is not None and self.dataset_cfg.name == 'schoffelen2019'

        for idx, row in unique_recordings.iterrows():
            key = (row.subject, row.session, row.task)
            self.recording_index_map[key] = idx

            if is_mous:
                recordings_str = f"Schoffelen2019Recording('{row.subject}_{row.task}')"
            else:
                story_num = story2task(row.task)
                recordings_str = f"Gwilliams2022Recording('{format_sub(row.subject)}_session{row.session}_story{story_num}')"
            self.recordings_map[key] = recordings_str

    def _baseline_correct(self, data: np.ndarray, sfreq: float, baseline_dur: float = 0.5) -> np.ndarray:
        """Apply baseline correction to MEG data"""

        n_baseline_samples = int(round(baseline_dur * sfreq))
        if n_baseline_samples > data.shape[1]:
            raise ValueError(f"Not enough samples for baseline correction: required {n_baseline_samples}, available {data.shape[1]}")

        if n_baseline_samples > 0:
            baseline = data[:, :n_baseline_samples].mean(axis=1, keepdims=True)
            data = data - baseline

        return data

    # ---------- MEG memmap methods ----------
    def _meg_cache_files(self, row) -> Tuple[Path, Path]:
        """Generate cache file paths for a MEG recording."""
        if self.dataset_cfg is not None and self.dataset_cfg.name == 'schoffelen2019':
            parts = [f"sub-{row.subject}", f"task-{row.task}"]
            if self.dataset_cfg.run_column:
                run_val = getattr(row, self.dataset_cfg.run_column, None)
                if run_val is not None and pd.notna(run_val):
                    parts.append(f"run-{int(run_val)}")
            key = '_'.join(parts)
        else:
            key = '_'.join([
                f"sub-{format_sub(row.subject)}",
                f"ses-{str(row.session)}",
                f"task-{story2task(row.task)}"
            ])
        data = self.meg_cfg.cache_dir / f"{key}.npy"
        meta = self.meg_cfg.cache_dir / f"{key}.json"
        return data, meta

    def _create_meg_memmap(self, row) -> Tuple[np.memmap, float, int]:
        """Create memmap cache for MEG data."""
        import mne
        from mne_bids import BIDSPath, read_raw_bids

        data_file, meta_file = self._meg_cache_files(row)

        # Build BIDSPath — dataset-specific
        if self.dataset_cfg is not None and self.dataset_cfg.name == 'schoffelen2019':
            run_val = None
            if self.dataset_cfg.run_column:
                rv = getattr(row, self.dataset_cfg.run_column, None)
                if rv is not None and pd.notna(rv):
                    run_val = str(int(rv))
            bp = BIDSPath(
                root=self.meg_cfg.bids_root,
                subject=str(row.subject),
                datatype="meg",
                task=str(row.task),
                run=run_val,
            )
        else:
            bp = BIDSPath(
                root=self.meg_cfg.bids_root,
                subject=format_sub(row.subject),
                session=str(row.session),
                datatype="meg",
                task=story2task(row.task),
            )
        try:
            raw = read_raw_bids(bp, verbose="ERROR")
        except (ValueError, FileNotFoundError):
            # Broken _scans.tsv or missing BIDS entity — fall back to raw CTF read
            ds_files = sorted(bp.directory.glob(f"{bp.basename}*_meg.ds"))
            if ds_files:
                raw = mne.io.read_raw_ctf(str(ds_files[0]), preload=False, verbose="ERROR")
            else:
                raise

        # Pick channels before resampling
        if self.meg_cfg.picks is not None:
            picks = mne.pick_channels_regexp(raw.info['ch_names'], regexp=self.meg_cfg.picks)
        else:
            ref_meg = self.dataset_cfg.meg_ref_meg if self.dataset_cfg is not None else False
            picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ecg=False,
                                   stim=False, ref_meg=ref_meg, exclude="bads")

        # Exclude bad channels (strip suffix before comparing, e.g. "MLC11-4304" → "MLC11")
        bad = set(self.dataset_cfg.meg_exclude) if self.dataset_cfg else set()
        if bad:
            picks = [p for p in picks if raw.info['ch_names'][p].rsplit('-', 1)[0] not in bad]

        raw.pick(picks)

        # Resample
        if abs(raw.info["sfreq"] - self.meg_cfg.target_hz) > 1e-6:
            raw.resample(self.meg_cfg.target_hz, verbose="ERROR")

        # Get data
        data = raw.get_data().astype(np.float32)  # [C, T]
        n_channels, n_times = data.shape
        sfreq = self.meg_cfg.target_hz

        # Create memmap
        memmap_data = np.memmap(data_file, dtype=np.float32, mode='w+', shape=data.shape)
        memmap_data[:] = data[:]
        memmap_data.flush()

        # Save lightweight metadata
        metadata = {'shape': [n_channels, n_times],'sfreq': sfreq,'dtype': 'float32'}
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)

        # Close raw file
        raw.close()

        return memmap_data, sfreq, n_channels

    def _get_meg_memmap(self, row) -> Tuple[np.memmap, float, int]:
        """Get MEG memmap data, creating cache if needed."""
        data_file, meta_file = self._meg_cache_files(row)

        if data_file.exists() and meta_file.exists():
            # Load metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)

            shape = tuple(metadata['shape'])
            sfreq = metadata['sfreq']
            n_channels = shape[0]

            # Load memmap
            memmap_data = np.memmap(data_file, dtype=np.float32, mode='r', shape=shape)
            return memmap_data, sfreq, n_channels
        else:
            return self._create_meg_memmap(row)

    # ---------- Audio memmap methods ----------
    def _audio_path(self, row) -> Path:
        return (Path(self.audio_cfg.stim_root) / str(row.audio_file)).resolve()

    def _audio_cache_files(self, audio_file: str) -> str:
        """Generate cache key for audio data."""
        filename = Path(audio_file).stem
        key = f"{filename}_{self.audio_cfg.resample_hz}hz"
        data = self.audio_cfg.cache_dir / f"{key}.npy"
        meta = self.audio_cfg.cache_dir / f"{key}.json"
        return data, meta

    def _create_audio_memmap(self, audio_path: Path) -> Tuple[np.memmap, float]:
        """Create memmap cache for audio data."""
        import librosa
        import soundfile as sf

        data_file, meta_file = self._audio_cache_files(audio_path.name)

        # Load and process audio using soundfile directly
        y, sr = sf.read(str(audio_path), always_2d=True)

        # Convert to mono if stereo (y is [time, channels] from soundfile)
        if y.shape[1] > 1:
            y = np.mean(y, axis=1)
        else:
            y = y.squeeze()

        # Resample
        if abs(sr - self.audio_cfg.resample_hz) > 1e-6:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.audio_cfg.resample_hz)

        y = y.astype(np.float32)  # [time]
        n_samples = len(y)
        sfreq = self.audio_cfg.resample_hz

        # Create memmap
        memmap_data = np.memmap(data_file, dtype=np.float32, mode='w+', shape=y.shape)
        memmap_data[:] = y[:]
        memmap_data.flush()

        # Save lightweight metadata
        metadata = {'shape': [n_samples],'sfreq': sfreq,'dtype': 'float32'}
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)

        return memmap_data, sfreq

    def _get_audio_memmap(self, audio_path: Path) -> Tuple[np.memmap, float]:
        """Get audio memmap data, creating cache if needed."""
        data_file, meta_file = self._audio_cache_files(audio_path.name)

        if data_file.exists() and meta_file.exists():
            # Load metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)

            shape = tuple(metadata['shape'])
            sfreq = metadata['sfreq']

            # Load memmap with correct shape
            memmap_data = np.memmap(data_file, dtype=np.float32, mode='r', shape=shape)
            return memmap_data, sfreq
        else:
            return self._create_audio_memmap(audio_path)

    # ---------- Data slicing methods ----------

    def _slice_meg(self, memmap_data: np.memmap, sfreq: float, t0: float, dur: float, baseline_correct: bool = True) -> Tuple[np.ndarray, float]:
        """Slice MEG data from memmap."""
        n_samples = int(round(dur * sfreq))
        start_idx = int(round(t0 * sfreq))
        end_idx = start_idx + n_samples

        # Ensure we don't exceed array size
        start_idx = max(0, start_idx)
        end_idx = min(memmap_data.shape[1], end_idx)

        data = memmap_data[:, start_idx:end_idx].copy()

        # Pad if necessary
        if data.shape[1] < n_samples:
            pad_width = ((0, 0), (0, n_samples - data.shape[1]))
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
        elif data.shape[1] > n_samples:
            data = data[:, :n_samples]

        # Apply baseline correction
        if baseline_correct:
            data = self._baseline_correct(data, sfreq)

        return data, sfreq

    def _slice_audio(self, memmap_data: np.memmap, sfreq: float, start: float, duration: float) -> Tuple[np.ndarray, float]:
        """Slice audio data from memmap."""
        n_samples = int(round(duration * sfreq))
        s0 = int(round(start * sfreq))
        s1 = s0 + n_samples

        # Ensure bounds
        s0 = max(0, s0)
        s1 = min(len(memmap_data), s1)

        # Slice and copy
        y_seg = memmap_data[s0:s1].copy()

        # Pad if necessary
        if len(y_seg) < n_samples:
            pad = np.zeros(n_samples - len(y_seg), dtype=y_seg.dtype)
            y_seg = np.concatenate([y_seg, pad], axis=0)
        elif len(y_seg) > n_samples:
            y_seg = y_seg[:n_samples]

        return y_seg, sfreq

    # ---------- PyTorch ----------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Get MEG data
        memmap_meg, meg_sfreq, n_channels = self._get_meg_memmap(row)
        meg, meg_sr = self._slice_meg(memmap_meg, meg_sfreq, row.onset, row.duration, baseline_correct=True)

        # Get audio data
        audio_path = self._audio_path(row)
        memmap_audio, audio_sfreq = self._get_audio_memmap(audio_path)
        wav, wav_sr = self._slice_audio(memmap_audio, audio_sfreq, row.start, row.duration)

        # Force same length if needed (default False)
        if self.align_cfg.trim_to_min:
            meg, wav = _trim_to_min(meg, wav, axis=1)

        # Get recording mappings
        recording_key = (row.subject, row.session, row.task)
        recording_index = self.recording_index_map[recording_key]
        recordings_str = self.recordings_map[recording_key]

        return {
            "meg": torch.tensor(meg),             # [C, T]
            "features": torch.tensor(wav),     # [T]
            "words": str(row.words),
            "sr": meg_sr,
            "wav_sr": wav_sr,
            "subject": torch.tensor(self.subject_index_map[row.subject], dtype=torch.int64),
            "session": str(row.session),
            "task": str(row.task),
            "audio_file": str(row.audio_file),
            "onset": float(row.onset),
            "start": float(row.start),
            "duration": float(row.duration),
            "anchor_word": str(row.anchor_word),
            "block_uid": str(row.block_uid),
            "block_stim_uid": str(row.block_stim_uid),
            "segment_uid": str(row.segment_uid),
            "stim_id": str(row.stim_id),
            "speech_rate": float(getattr(row, 'speech_rate', 0.0)),
            "condition": str(row.condition),
            "recording_index": int(recording_index),
            "recordings": recordings_str
        }