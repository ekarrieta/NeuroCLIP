"""Prepare MEG-MASC word annotations for NeuroCLIP."""

from __future__ import annotations

import argparse
import ast
import hashlib
import logging
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from readers.dataset import DatasetConfig
from utils import PROJECT_ROOT, configure_logging


TASK_NAMES = {
    "0": "lw1",
    "1": "cable_spool_fort",
    "2": "easy_money",
    "3": "the_black_willow",
}
SPLIT_NAMES = {0: "test", 1: "valid", 2: "train"}


def parse_description(value: Any) -> dict[str, Any]:
    """Parse the stringified dictionary stored in a BIDS trial_type field."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    value = value.strip()
    if not (value.startswith("{") and value.endswith("}")):
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_audio_name(value: Any) -> str | None:
    """Return a portable audio basename and normalize numeric .0 suffixes."""
    if not isinstance(value, str) or not value:
        return None
    name = Path(value).name
    match = re.match(
        r"^(.*?_)(\d+(?:\.\d+)?)\.wav$", name, re.IGNORECASE
    )
    if match:
        prefix, number = match.groups()
        return f"{prefix}{int(float(number))}.wav".lower()
    return name.lower()


def _entity(filename: str, key: str) -> str:
    match = re.search(rf"(?:^|_){key}-([^_]+)", filename)
    if match is None:
        raise ValueError(f"Cannot read {key!r} from {filename}")
    return match.group(1)


def extract_word_events(bids_root: Path) -> pd.DataFrame:
    """Read MEG-MASC BIDS event files into a normalized word-event table."""
    event_files = sorted(
        bids_root.glob("sub-*/ses-*/meg/*_events.tsv")
    )
    if not event_files:
        raise FileNotFoundError(
            f"No BIDS MEG event files found below {bids_root}"
        )

    rows: list[dict[str, Any]] = []
    for event_path in event_files:
        subject = _entity(event_path.name, "sub")
        session = _entity(event_path.name, "ses")
        task_id = _entity(event_path.name, "task")
        events = pd.read_csv(event_path, sep="\t")

        for event in events.to_dict("records"):
            description = parse_description(event.get("trial_type"))
            if description.get("kind") != "word":
                continue
            audio_file = normalize_audio_name(
                description.get("sound") or description.get("audio")
            )
            word_index = description.get("word_index")
            if audio_file is None or word_index is None:
                continue
            speech_rate = description.get("speech_rate")
            rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "task": TASK_NAMES.get(task_id, task_id),
                    "audio_file": audio_file,
                    "onset": float(event["onset"]),
                    "start": round(float(description["start"]), 5),
                    "duration": round(float(event["duration"]), 5),
                    "word": str(description.get("word", "")),
                    "word_index": int(word_index),
                    "condition": str(
                        description.get("condition", "")
                    ),
                    "speech_rate": (
                        float(speech_rate)
                        if speech_rate is not None
                        else np.nan
                    ),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No word events could be parsed.")
    return result.sort_values(
        ["subject", "session", "task", "onset"]
    ).reset_index(drop=True)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _content_split(
    text: str, seed: int, valid_ratio: float, test_ratio: float
) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = random.Random(seed + int.from_bytes(digest[:8], "big"))
    score = rng.random()
    if score < test_ratio:
        return 0
    if score < test_ratio + valid_ratio:
        return 1
    return 2


def create_blocks(
    events: pd.DataFrame,
    seed: int,
    valid_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    """Group consecutive word events into sentence or audio blocks."""
    blocks: list[dict[str, Any]] = []
    keys = ["subject", "session", "task"]
    for _, group in events.groupby(keys, sort=False):
        group = group.sort_values("onset").reset_index(drop=True)
        starts = group.index[group["word_index"] == 0].tolist()
        for position, start_index in enumerate(starts):
            start_row = group.loc[start_index]
            next_index = (
                starts[position + 1]
                if position + 1 < len(starts)
                else len(group)
            )
            block_events = group.iloc[start_index:next_index]
            if block_events.empty:
                continue
            block_start = float(start_row["onset"])
            block_end = float(
                (
                    block_events["onset"]
                    + block_events["duration"]
                ).max()
            )
            words = " ".join(block_events["word"].astype(str))
            start_ms = int(round(float(start_row["start"]) * 1000))
            content_key = (
                f"{start_row['task']}|{start_row['audio_file']}|"
                f"{start_ms}|{words}"
            )
            block_uid = _short_hash(
                f"{start_row['subject']}|{start_row['session']}|"
                f"{content_key}|{block_start:.5f}"
            )
            blocks.append(
                {
                    "subject": str(start_row["subject"]),
                    "session": str(start_row["session"]),
                    "task": str(start_row["task"]),
                    "audio_file": str(start_row["audio_file"]),
                    "onset": block_start,
                    "start": float(start_row["start"]),
                    "duration": block_end - block_start,
                    "words": words,
                    "block_uid": block_uid,
                    "block_stim_uid": _short_hash(content_key),
                    "speech_rate": start_row["speech_rate"],
                    "condition": str(start_row["condition"]),
                    "split": _content_split(
                        words, seed, valid_ratio, test_ratio
                    ),
                }
            )
    return pd.DataFrame(blocks)


def create_segments(
    events: pd.DataFrame,
    blocks: pd.DataFrame,
    tmin: float,
    tmax: float,
) -> pd.DataFrame:
    """Create word-anchored windows contained within their source block."""
    segments: list[dict[str, Any]] = []
    keys = ["subject", "session", "task", "audio_file"]
    event_groups = {
        key: value.sort_values("onset")
        for key, value in events.groupby(keys, sort=False)
    }

    for block in blocks.to_dict("records"):
        key = tuple(block[column] for column in keys)
        block_events = event_groups[key]
        block_start = float(block["onset"])
        block_end = block_start + float(block["duration"])
        block_events = block_events[
            (block_events["onset"] >= block_start)
            & (block_events["onset"] <= block_end)
        ]

        for word in block_events.to_dict("records"):
            segment_start = float(word["onset"]) + tmin
            segment_end = float(word["onset"]) + tmax
            if segment_start < block_start or segment_end > block_end:
                continue
            window = block_events[
                (block_events["onset"] >= segment_start)
                & (
                    block_events["onset"]
                    + block_events["duration"]
                    <= segment_end
                )
            ]
            audio_start = float(word["start"]) + tmin
            stimulus_key = (
                f"{word['task']}|{word['audio_file']}|"
                f"{int(round(audio_start * 1000))}|{word['word']}"
            )
            segments.append(
                {
                    "subject": str(word["subject"]),
                    "session": str(word["session"]),
                    "task": str(word["task"]),
                    "audio_file": str(word["audio_file"]),
                    "onset": round(segment_start, 6),
                    "start": round(audio_start, 6),
                    "duration": round(tmax - tmin, 6),
                    "split": int(block["split"]),
                    "words": " ".join(window["word"].astype(str)),
                    "anchor_word": str(word["word"]),
                    "block_uid": str(block["block_uid"]),
                    "block_stim_uid": str(
                        block["block_stim_uid"]
                    ),
                    "segment_uid": _short_hash(
                        f"{block['block_uid']}|"
                        f"{segment_start:.6f}|{segment_end:.6f}"
                    ),
                    "stim_id": _short_hash(stimulus_key),
                    "speech_rate": word["speech_rate"],
                    "condition": str(block["condition"]),
                }
            )

    return pd.DataFrame(segments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare MEG-MASC annotations for NeuroCLIP."
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=PROJECT_ROOT
        / "readers"
        / "gwilliams2022.yaml",
    )
    parser.add_argument("--bids-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "cache"
        / "gwilliams2022"
        / "annotations",
    )
    parser.add_argument(
        "--lengths", type=float, nargs="+", default=[3.0]
    )
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--valid-ratio", type=float, default=0.17)
    parser.add_argument("--test-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if any(length <= 0 for length in args.lengths):
        raise ValueError("All segment lengths must be positive.")
    if (
        args.valid_ratio < 0
        or args.test_ratio < 0
        or args.valid_ratio + args.test_ratio >= 1
    ):
        raise ValueError(
            "Validation and test ratios must be nonnegative "
            "and sum to less than one."
        )

    configure_logging()
    dataset_cfg = DatasetConfig.from_yaml(args.dataset_config)
    bids_root = (
        args.bids_root.resolve()
        if args.bids_root is not None
        else dataset_cfg.bids_root
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = extract_word_events(bids_root)
    events.to_csv(output_dir / "events.tsv", sep="\t", index=False)
    blocks = create_blocks(
        events, args.seed, args.valid_ratio, args.test_ratio
    )
    blocks.to_csv(output_dir / "blocks.tsv", sep="\t", index=False)
    logging.info(
        "Parsed %d word events into %d blocks", len(events), len(blocks)
    )

    for length in args.lengths:
        segments = create_segments(
            events, blocks, args.tmin, args.tmin + length
        )
        output_path = output_dir / f"segments_{length:.1f}s.tsv"
        segments.to_csv(output_path, sep="\t", index=False)
        counts = {
            SPLIT_NAMES[int(split)]: int(count)
            for split, count in segments["split"].value_counts().items()
        }
        logging.info(
            "Saved %d %.1fs segments to %s (%s)",
            len(segments),
            length,
            output_path,
            counts,
        )


if __name__ == "__main__":
    main()
