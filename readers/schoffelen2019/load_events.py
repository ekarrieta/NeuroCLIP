"""
MOUS (Schoffelen 2019) — Phase 1: Event extraction
====================================================
Parses the BIDS events.tsv files for all auditory MEG subjects and produces a
single events.tsv with one row per trial (sentence or word-list presentation).

Output columns
--------------
subject       : str  — BIDS subject ID without "sub-" prefix, e.g. "A2002"
session       : str  — always "0" (MOUS has no repeated MEG sessions)
task          : str  — always "auditory"
run           : int or NaN — 1 or 2 for multi-run subjects, NaN otherwise
stimulus_id   : int  — numeric ID of the wav file (maps into stimuli.txt)
audio_file    : str  — wav filename, e.g. "EQ_Ramp_Int2_Int1LPF186.wav"
sound_onset   : float — MEG time (s) of trigger-14 "Start File" event
audio_onset   : float — MEG time (s) of "Audio onset" event (first word)
target_onset  : float — MEG time (s) of "target" event
end_onset     : float — MEG time (s) of "End of file" event
condition     : str  — "sentence" or "wordlist"
complexity    : str  — "RC+" or "RC-"
words         : str  — full stimulus text from stimuli.txt

Usage
-----
    python readers/schoffelen2019/load_events.py

Output is written to readers/schoffelen2019/events.tsv.
"""

from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
BIDS_ROOT   = REPO_ROOT / 'data' / 'schoffelen2019'
STIMULI_TXT = BIDS_ROOT / 'stimuli' / 'stimuli.txt'
OUT_PATH    = Path(__file__).parent / 'events.tsv'

# ── Constants ─────────────────────────────────────────────────────────────────
# Subjects for whom the MEG task is split into two runs (acquisition crash)
MULTI_RUN_SUBJECTS = {'A2011', 'A2036', 'A2062', 'A2063', 'A2076', 'A2084'}

# Trigger codes that appear in "Nothing" events
# code → (condition, complexity)
ONSET_CODE_MAP = {
    1: ('sentence', 'RC+'),
    3: ('wordlist', 'RC+'),
    5: ('sentence', 'RC-'),
    7: ('wordlist', 'RC-'),
}
TARGET_CODES = {2, 4, 6, 8}

# Regex to extract stimulus number from Sound value "14 Start File 186.wav"
RE_SOUND = re.compile(r'Start File (\d+)\.wav', re.IGNORECASE)

# Audio file template
AUDIO_TEMPLATE = 'EQ_Ramp_Int2_Int1LPF{:03d}.wav'


# ── Load stimulus text lookup ─────────────────────────────────────────────────

def load_stimuli(path: Path) -> dict[int, str]:
    """Return {stimulus_id: text} from stimuli.txt (format: "NNN text...")."""
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


# ── Per-subject event parsing ─────────────────────────────────────────────────

def parse_events_file(events_path: Path) -> pd.DataFrame:
    """Parse a single *_events.tsv and return it as a DataFrame."""
    df = pd.read_csv(events_path, sep='\t', dtype=str)
    df['onset'] = df['onset'].astype(float)
    return df


def extract_trials(events_df: pd.DataFrame) -> list[dict]:
    """
    Extract one trial record per Sound event.

    A trial consists of:
      Sound     → "14 Start File NNN.wav"       (wav playback starts)
      Nothing   → "{1|3|5|7} Audio onset"        (first word, most accurate timing)
      Nothing   → "{2|4|6|8} target"             (target word)
      Nothing   → "15 End of file"               (wav playback ends)
    """
    rows = events_df.to_dict('records')
    trials = []

    i = 0
    while i < len(rows):
        row = rows[i]

        # Look for Sound events that encode a stimulus file
        if row['type'] != 'Sound':
            i += 1
            continue
        m = RE_SOUND.search(str(row.get('value', '')))
        if m is None:
            i += 1
            continue

        stimulus_id = int(m.group(1))
        sound_onset = row['onset']
        audio_file  = AUDIO_TEMPLATE.format(stimulus_id)

        # Scan forward for Audio onset, target, and End of file
        audio_onset  = None
        target_onset = None
        end_onset    = None
        condition    = None
        complexity   = None

        j = i + 1
        while j < len(rows):
            nrow = rows[j]
            ntype  = nrow['type']
            nvalue = str(nrow.get('value', ''))

            if ntype == 'Sound':        # next stimulus starts — stop looking
                break

            if ntype == 'Nothing':
                # "Audio onset" events
                ao_match = re.match(r'^(\d+)\s+Audio onset', nvalue)
                if ao_match:
                    code = int(ao_match.group(1))
                    if code in ONSET_CODE_MAP:
                        audio_onset = nrow['onset']
                        condition, complexity = ONSET_CODE_MAP[code]

                # "target" events
                tgt_match = re.match(r'^(\d+)\s+target', nvalue)
                if tgt_match:
                    code = int(tgt_match.group(1))
                    if code in TARGET_CODES:
                        target_onset = nrow['onset']

                # "End of file"
                if re.match(r'^15\s+End of file', nvalue):
                    end_onset = nrow['onset']
                    break   # trial is complete

            j += 1

        # Only keep trials where we have the minimum timing info
        if audio_onset is not None and end_onset is not None:
            trials.append({
                'stimulus_id':  stimulus_id,
                'audio_file':   audio_file,
                'sound_onset':  sound_onset,
                'audio_onset':  audio_onset,
                'target_onset': target_onset,
                'end_onset':    end_onset,
                'condition':    condition,
                'complexity':   complexity,
            })

        i = j  # resume from where we stopped

    return trials


def get_events_files(subject: str, bids_root: Path) -> list[tuple[Path, int | None]]:
    """
    Return list of (events_path, run) pairs for a subject.
    run=None for single-run subjects.
    """
    meg_dir = bids_root / f'sub-{subject}' / 'meg'
    if subject in MULTI_RUN_SUBJECTS:
        files = []
        for run in (1, 2):
            p = meg_dir / f'sub-{subject}_task-auditory_run-{run}_events.tsv'
            if p.exists():
                files.append((p, run))
        return files
    else:
        p = meg_dir / f'sub-{subject}_task-auditory_events.tsv'
        if p.exists():
            return [(p, None)]
        return []


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f'Loading stimulus texts from {STIMULI_TXT}')
    stimuli = load_stimuli(STIMULI_TXT)
    print(f'  Loaded {len(stimuli)} stimulus entries')

    # Get auditory subjects from participants.tsv
    participants = pd.read_csv(BIDS_ROOT / 'participants.tsv', sep='\t')
    auditory_subs = sorted(
        p.replace('sub-', '')
        for p in participants['participant_id']
        if p.startswith('sub-A')
    )
    print(f'Found {len(auditory_subs)} auditory subjects')

    all_rows = []
    missing  = []

    for subject in auditory_subs:
        event_files = get_events_files(subject, BIDS_ROOT)

        if not event_files:
            missing.append(subject)
            continue

        for events_path, run in event_files:
            events_df = parse_events_file(events_path)
            trials    = extract_trials(events_df)

            for t in trials:
                sid = t['stimulus_id']
                all_rows.append({
                    'subject':      subject,
                    'session':      '0',      # MOUS has no repeated sessions
                    'task':         'auditory',
                    'run':          run,      # NaN for single-run subjects
                    'stimulus_id':  sid,
                    'audio_file':   t['audio_file'],
                    'sound_onset':  t['sound_onset'],
                    'audio_onset':  t['audio_onset'],
                    'target_onset': t['target_onset'],
                    'end_onset':    t['end_onset'],
                    'condition':    t['condition'],
                    'complexity':   t['complexity'],
                    'words':        stimuli.get(sid, ''),
                })

        n = len([r for r in all_rows if r['subject'] == subject])
        runs_str = f'  ({len(event_files)} run(s))' if subject in MULTI_RUN_SUBJECTS else ''
        print(f'  sub-{subject}{runs_str}: {n} trials')

    df = pd.DataFrame(all_rows)

    print(f'\nTotal: {len(df)} trials across {df["subject"].nunique()} subjects')
    print(f'Missing events files: {missing}')

    cond_counts = df['condition'].value_counts()
    print(f'Condition counts:\n{cond_counts.to_string()}')

    empty_words = (df['words'] == '').sum()
    if empty_words > 0:
        print(f'WARNING: {empty_words} trials have no stimulus text (unknown stimulus_id)')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, sep='\t', index=False)
    print(f'\nSaved to {OUT_PATH}')


if __name__ == '__main__':
    main()
