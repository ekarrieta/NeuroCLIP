"""
MOUS — Phase 2a: Prepare audio + transcripts for Montreal Forced Aligner (MFA)
===============================================================================
Creates the directory structure required by MFA to run Dutch forced alignment
on all 808 MOUS audio files.

MFA expects:
    mfa_input/
        {speaker}/
            {utterance}.wav      ← audio file (mono, any sr)
            {utterance}.lab      ← plain-text transcript

We use one "speaker" per file (speaker = stimulus ID) so each file is
treated independently. This avoids any cross-stimulus coarticulation issues.

After running this script, execute:
    mfa align mfa_input/ dutch_mfa dutch_mfa mfa_output/ --clean

(Requires: `pip install montreal-forced-aligner`
           `mfa model download acoustic dutch_mfa`
           `mfa model download dictionary dutch_mfa`)

The output TextGrids will be at:
    mfa_output/{speaker}/{utterance}.TextGrid

Then run parse_alignments.py to convert TextGrids → JSON.
"""

from __future__ import annotations
from pathlib import Path

import soundfile as sf
import numpy as np
import librosa

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
BIDS_ROOT   = REPO_ROOT / 'data' / 'schoffelen2019'
AUDIO_DIR   = BIDS_ROOT / 'stimuli' / 'audio_files'
STIMULI_TXT = BIDS_ROOT / 'stimuli' / 'stimuli.txt'
MFA_INPUT   = Path(__file__).parent / 'mfa_input'
MFA_INPUT.mkdir(exist_ok=True)


def load_stimuli(path: Path) -> dict[int, str]:
    stimuli = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2 and parts[0].isdigit():
                stimuli[int(parts[0])] = parts[1]
    return stimuli


def load_mono_16k(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio, convert to mono float32, resample to 16 kHz."""
    y, sr = sf.read(str(audio_path), always_2d=True)
    if y.shape[1] > 1:
        y = np.mean(y, axis=1)
    else:
        y = y.squeeze()
    if sr != 16000:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
    return y.astype(np.float32), 16000


def main():
    stimuli = load_stimuli(STIMULI_TXT)
    print(f'Loaded {len(stimuli)} stimulus texts')

    audio_files = sorted(AUDIO_DIR.glob('EQ_Ramp_Int2_Int1LPF*.wav'))
    print(f'Found {len(audio_files)} audio files')

    prepared = skipped = 0
    for audio_path in audio_files:
        # Extract stimulus ID from filename
        stem = audio_path.stem                          # EQ_Ramp_Int2_Int1LPF186
        stim_id = int(stem.replace('EQ_Ramp_Int2_Int1LPF', ''))

        text = stimuli.get(stim_id)
        if not text:
            print(f'  WARNING: no text for stimulus {stim_id}, skipping')
            skipped += 1
            continue

        # MFA directory: mfa_input/{stim_id}/
        speaker_dir = MFA_INPUT / str(stim_id)
        speaker_dir.mkdir(exist_ok=True)

        wav_out = speaker_dir / f'{stim_id}.wav'
        lab_out = speaker_dir / f'{stim_id}.lab'

        # Write transcript (MFA .lab format = plain UTF-8 text, one line)
        lab_out.write_text(text, encoding='utf-8')

        # Write mono 16 kHz wav (skip if already done)
        if not wav_out.exists():
            y, sr = load_mono_16k(audio_path)
            import soundfile as sf
            sf.write(str(wav_out), y, sr, subtype='PCM_16')

        prepared += 1

    print(f'\nPrepared {prepared} files, skipped {skipped}')
    print(f'MFA input directory: {MFA_INPUT}')
    print(
        '\nNext step — run forced alignment:\n'
        f'  mfa align {MFA_INPUT} dutch_mfa dutch_mfa {Path(__file__).parent / "mfa_output"} --clean\n'
        '\nThen run:  python readers/schoffelen2019/parse_alignments.py'
    )


if __name__ == '__main__':
    main()
