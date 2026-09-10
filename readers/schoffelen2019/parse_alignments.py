"""
MOUS — Phase 2b: Parse MFA TextGrid outputs → per-stimulus JSON alignments
===========================================================================
Reads the TextGrid files produced by Montreal Forced Aligner and converts
each one to a compact JSON file with word-level onset/offset times (relative
to the start of the audio file).

Output JSON format (one file per stimulus):
    alignments/{stim_id}.json
    [
        {"word": "De",    "onset": 0.124, "offset": 0.234},
        {"word": "bands", "onset": 0.280, "offset": 0.510},
        ...
    ]

The "sp" (short pause) intervals emitted by MFA are excluded.

Validation
----------
Reports the distribution of MFA first-word onset times (seconds relative to
wav file start).  Expected range: 0–0.1 s (small leading silence, if any).

Note: the presentation code's first_word_onset_ms is relative to *trial onset*
(including fixation cross, ~2–3 s), NOT to the wav file start, so it cannot be
used as a ground truth here.

Usage
-----
    python readers/schoffelen2019/parse_alignments.py
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
MFA_OUTPUT  = Path(__file__).parent / 'mfa_output'
ALIGN_DIR   = Path(__file__).parent / 'alignments'
ALIGN_DIR.mkdir(exist_ok=True)


# ── TextGrid parser ───────────────────────────────────────────────────────────

def parse_textgrid(path: Path) -> list[dict]:
    """
    Parse an MFA TextGrid file and return word intervals.
    Returns list of {'word': str, 'onset': float, 'offset': float}.
    Skips silence/pause entries ('sp', 'sil', '').
    """
    text = path.read_text(encoding='utf-8')

    # Find the "words" tier (IntervalTier named "words")
    # MFA TextGrid format — locate the tier block
    tier_blocks = re.split(r'item\s*\[\d+\]', text)

    words_block = None
    for block in tier_blocks:
        if '"words"' in block or '"word"' in block:
            words_block = block
            break

    if words_block is None:
        return []

    # Extract all intervals
    intervals = re.findall(
        r'xmin\s*=\s*([\d.]+)\s+xmax\s*=\s*([\d.]+)\s+text\s*=\s*"([^"]*)"',
        words_block,
        re.MULTILINE,
    )

    result = []
    for xmin_s, xmax_s, word in intervals:
        word = word.strip()
        if word in ('', 'sp', 'sil', 'SIL', 'SP'):
            continue
        result.append({
            'word':   word,
            'onset':  round(float(xmin_s), 4),
            'offset': round(float(xmax_s), 4),
        })
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tg_files = sorted(MFA_OUTPUT.rglob('*.TextGrid'))
    print(f'Found {len(tg_files)} TextGrid files in {MFA_OUTPUT}')

    if not tg_files:
        print('No TextGrid files found. Run MFA first (see prepare_alignment.py).')
        return

    first_onsets_ms = []
    n_ok = n_empty = 0

    for tg_path in tg_files:
        # stimulus ID is the parent directory name
        try:
            stim_id = int(tg_path.parent.name)
        except ValueError:
            continue

        words = parse_textgrid(tg_path)
        if not words:
            n_empty += 1
            print(f'  WARNING: empty alignment for stimulus {stim_id}')
            continue

        # Write JSON
        out_path = ALIGN_DIR / f'{stim_id}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False)
        n_ok += 1

        first_onsets_ms.append(words[0]['onset'] * 1000.0)

    print(f'\nParsed: {n_ok} OK, {n_empty} empty')
    if first_onsets_ms:
        arr = np.array(first_onsets_ms)
        print(f'MFA first-word onset (ms from wav start) — expected 0–100 ms:')
        print(f'  mean={arr.mean():.1f}ms  std={arr.std():.1f}ms  '
              f'min={arr.min():.1f}ms  max={arr.max():.1f}ms  (n={len(arr)})')
        n_suspicious = int(np.sum(arr > 200))
        if n_suspicious:
            print(f'  WARNING: {n_suspicious} stimuli have first-word onset > 200ms '
                  f'(may indicate alignment issues)')
    print(f'\nAlignments saved to {ALIGN_DIR}')


if __name__ == '__main__':
    main()
