# Dataset preparation

NeuroCLIP expects a BIDS dataset and a tab-separated segment table. The
built-in dataset YAML files define the default BIDS and stimulus locations.
Relative paths are resolved from the directory selected by path_base.

The segment table has one row per word-anchored window. Required columns are
subject, session, task, audio_file, onset, start, duration, split, words,
anchor_word, block_uid, block_stim_uid, segment_uid, stim_id, and condition.
Split values are 0 for test, 1 for validation, and 2 for training.
Content-level identifiers and splits are shared by subjects who heard the same
stimulus, preventing the same speech segment from crossing splits.

## MEG-MASC / Gwilliams 2022

Place the BIDS dataset at data/gwilliams2022, or pass --bids-root. Then run:

    python -m readers.gwilliams2022.prepare --lengths 3.0

The script reads word annotations from BIDS events files and writes events.tsv,
blocks.tsv, and segments_3.0s.tsv under cache/gwilliams2022/annotations.
Multiple window lengths can be generated in one pass, for example
--lengths 1.0 2.0 3.0.

Important options:

- --tmin: window start relative to the anchor word, default -0.5 seconds.
- --valid-ratio and --test-ratio: content-level split proportions.
- --seed: deterministic split seed.
- --output-dir: destination for generated tables.

## MOUS / Schoffelen 2019

Place the BIDS dataset at data/schoffelen2019. The auditory stimuli require
word-level forced alignments. Run the four phases from the repository root:

    python readers/schoffelen2019/load_events.py
    python readers/schoffelen2019/prepare_alignment.py
    mfa align readers/schoffelen2019/mfa_input dutch_mfa dutch_mfa readers/schoffelen2019/mfa_output --clean
    python readers/schoffelen2019/parse_alignments.py
    python readers/schoffelen2019/split_and_segment.py --tmin -0.5 --tmax 2.5

Montreal Forced Aligner is an external preprocessing dependency. Install it
separately and download its Dutch acoustic model and dictionary. The final
command writes cache/schoffelen2019/annotations/segments_3.0s.tsv.

Both pipelines create caches and annotations locally. Generated files are ignored by git.
