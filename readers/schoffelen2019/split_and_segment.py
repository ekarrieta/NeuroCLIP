"""
MOUS — Phase 3: Block + segment creation
=========================================
Reads events.tsv (Phase 1) and per-stimulus word alignments (Phase 2) to
produce word-level segments compatible with readers/dataset.py.

Concepts
--------
block     : one trial (sentence or word-list) for one subject.
            Identified by (subject, run, stimulus_id, audio_onset).
segment   : a [tmin, tmax] window centred on one word's audio onset.
            MEG time is referenced to the continuous MEG recording.
            Audio time is referenced to the start of the wav file.

Output columns (mirrors gwilliams2022/segments_*.tsv)
------------------------------------------------------
subject, session, task, run, audio_file, stimulus_id,
onset      : float — MEG time of segment start (= word_meg_time + tmin)
start      : float — wav-file time of segment start (= word_audio_onset + tmin)
duration   : float — segment length in seconds
words      : str   — space-separated words whose onsets fall in [onset, onset+dur]
anchor_word: str   — the word that triggered this segment
block_uid  : str   — subject-specific block hash (12 hex chars)
block_stim_uid: str — stimulus-level block hash (same across subjects, 12 hex)
segment_uid: str   — subject-specific segment hash (12 hex chars)
stim_id    : str   — stimulus-level segment hash (12 hex chars)
condition  : str   — "sentence" or "wordlist"
complexity : str   — "RC+" or "RC-"
split      : int   — 0=test, 1=valid, 2=train

Usage
-----
    python readers/schoffelen2019/split_and_segment.py [--tmin -0.5] [--tmax 2.5]

Requires:
    readers/schoffelen2019/events.tsv        (Phase 1 output)
    readers/schoffelen2019/alignments/*.json  (Phase 2 output)

Output:
    cache/schoffelen2019/annotations/segments_{duration}s.tsv
"""

from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
EVENTS_PATH  = HERE / 'events.tsv'
ALIGN_DIR    = HERE / 'alignments'
CACHE_ROOT   = Path('./cache/schoffelen2019/annotations')

# ── Split assignment ──────────────────────────────────────────────────────────
# Ratios must sum to 1.0.  Values: test=0, valid=1, train=2
SPLIT_RATIOS = {0: 0.10, 1: 0.20, 2: 0.70}


def _hex12(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def assign_split(stimulus_id: int) -> int:
    """Deterministic split assignment from stimulus_id hash."""
    h   = hashlib.sha256(str(stimulus_id).encode()).digest()
    rng = random.Random(int.from_bytes(h[:4], 'big'))
    r   = rng.random()
    cumulative = 0.0
    for split_val, ratio in sorted(SPLIT_RATIOS.items()):
        cumulative += ratio
        if r < cumulative:
            return split_val
    return 2  # fallback to train


def _block_uid(subject: str, run, stimulus_id: int, audio_onset: float) -> str:
    """Subject-specific block identifier."""
    run_str = str(run) if pd.notna(run) else 'none'
    return _hex12(f'{subject}_{run_str}_{stimulus_id}_{audio_onset:.4f}')


def _block_stim_uid(stimulus_id: int, audio_onset_in_file: float) -> str:
    """Subject-independent block identifier (same audio content = same stim uid)."""
    return _hex12(f'{stimulus_id}_{audio_onset_in_file:.4f}')


def _segment_uid(block_uid: str, seg_start: float, seg_end: float) -> str:
    return _hex12(f'{block_uid}_{seg_start:.4f}_{seg_end:.4f}')


def _stim_id(stimulus_id: int, anchor_word: str, word_onset_in_file: float) -> str:
    """Subject-independent segment identifier."""
    return _hex12(f'{stimulus_id}_{anchor_word}_{word_onset_in_file:.4f}')


# ── Alignment loading ─────────────────────────────────────────────────────────

def load_alignment(stimulus_id: int) -> list[dict] | None:
    """Return [{word, onset, offset}, ...] or None if not found."""
    p = ALIGN_DIR / f'{stimulus_id}.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ── Segment creation for one block ───────────────────────────────────────────

def make_segments(
    block: dict,
    alignment: list[dict],
    tmin: float,
    tmax: float,
) -> list[dict]:
    """
    Create word-level segments for a single block (trial).

    Parameters
    ----------
    block     : row from events.tsv (dict)
    alignment : list of {word, onset, offset} relative to wav file start
    tmin, tmax: segment window around each word onset (seconds)

    Returns
    -------
    List of segment dicts.
    """
    # MEG time of wav file start (trigger-14 onset)
    sound_meg = block['sound_onset']

    # Segment validity bounds: wav playback window in MEG time
    block_start_meg = block['sound_onset']  # wav file starts playing
    block_end_meg   = block['end_onset']

    segments = []
    duration = tmax - tmin

    for word_info in alignment:
        word_audio_onset  = word_info['onset']    # relative to wav start
        word_audio_offset = word_info['offset']

        # Map word audio onset to MEG time
        word_meg_onset = sound_meg + word_audio_onset

        # Segment time window (MEG)
        seg_start_meg = word_meg_onset + tmin
        seg_end_meg   = word_meg_onset + tmax

        # Reject segment if it falls outside the block boundaries
        if seg_start_meg < block_start_meg:
            continue
        if seg_end_meg > block_end_meg:
            continue

        # Audio start within wav file
        seg_start_wav = word_audio_onset + tmin

        # Collect all words whose onsets fall entirely within the segment
        seg_words = [
            w['word'] for w in alignment
            if seg_start_meg <= (sound_meg + w['onset']) < seg_end_meg
        ]

        run_val  = block.get('run')
        run_notna = run_val if pd.notna(run_val) else None

        b_uid     = _block_uid(block['subject'], run_val, block['stimulus_id'], block['audio_onset'])
        b_stim_uid = _block_stim_uid(block['stimulus_id'], 0.0)  # constant per stimulus
        s_uid     = _segment_uid(b_uid, seg_start_meg, seg_end_meg)
        s_stim_id = _stim_id(block['stimulus_id'], word_info['word'], word_audio_onset)

        segments.append({
            'subject':       block['subject'],
            'session':       block['session'],
            'task':          block['task'],
            'run':           run_notna,
            'audio_file':    block['audio_file'],
            'stimulus_id':   block['stimulus_id'],
            'onset':         round(seg_start_meg, 6),
            'start':         round(seg_start_wav, 6),
            'duration':      round(duration, 4),
            'words':         ' '.join(seg_words),
            'anchor_word':   word_info['word'],
            'block_uid':     b_uid,
            'block_stim_uid': b_stim_uid,
            'segment_uid':   s_uid,
            'stim_id':       s_stim_id,
            'condition':     block['condition'],
            'complexity':    block['complexity'],
            'split':         assign_split(block['stimulus_id']),
        })

    return segments


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmin', type=float, default=-0.5,
                        help='Segment start relative to word onset (s). Default: -0.5')
    parser.add_argument('--tmax', type=float, default=2.5,
                        help='Segment end relative to word onset (s). Default: 2.5')
    args = parser.parse_args()

    tmin, tmax = args.tmin, args.tmax
    duration   = tmax - tmin
    print(f'Segment window: [{tmin}, {tmax}] = {duration:.1f}s')

    # Load events
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f'{EVENTS_PATH} not found — run load_events.py first')
    events = pd.read_csv(EVENTS_PATH, sep='\t')
    print(f'Loaded {len(events)} trials from {len(events["subject"].unique())} subjects')

    # Sentence + wordlist stimulus IDs must land in the same split.
    # Sentences: IDs 1–408. Corresponding word lists: IDs 501–908 (sentence_id + 500).
    # We assign split based on the sentence stimulus_id (i.e., wordlist_id - 500 for word lists).
    def canonical_id(sid: int) -> int:
        return sid - 500 if sid >= 501 else sid

    all_segs = []
    n_no_align = 0

    for _, block in events.iterrows():
        sid       = int(block['stimulus_id'])
        alignment = load_alignment(sid)
        if alignment is None:
            n_no_align += 1
            continue

        segs = make_segments(block.to_dict(), alignment, tmin, tmax)

        # Override split with canonical ID to pair sentences with their word lists
        canonical = canonical_id(sid)
        split_val = assign_split(canonical)
        for s in segs:
            s['split'] = split_val

        all_segs.extend(segs)

    df = pd.DataFrame(all_segs)

    print(f'\n{"="*50}')
    print(f'Total segments : {len(df):,}')
    if len(df):
        print(f'Subjects       : {df["subject"].nunique()}')
        print(f'Stimuli        : {df["stimulus_id"].nunique()}')
        print(f'No alignment   : {n_no_align} blocks skipped')
        sc = df['condition'].value_counts()
        print(f'Condition:\n{sc.to_string()}')
        sp = df['split'].value_counts().sort_index()
        split_names = {0: 'test', 1: 'valid', 2: 'train'}
        for v, c in sp.items():
            print(f'  split {v} ({split_names[v]:5s}): {c:,}')

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_ROOT / f'segments_{duration:.1f}s.tsv'
    df.to_csv(out_path, sep='\t', index=False)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
