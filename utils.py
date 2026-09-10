"""Shared configuration, reproducibility, and output helpers."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_TYPES = (
    "wav2vec2",
    "flipped_wav2vec2",
    "wav",
    "mfcc",
    "mel_spectrogram",
    "vad",
    "envelope",
    "pitch",
    "rms",
    "zcr",
    "word_embeddings",
    "sentence_embeddings",
    "static_embeddings",
    "anchor_word",
    "noise",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and select deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(spec: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available and validate explicit devices."""
    resolved = "cuda" if spec == "auto" and torch.cuda.is_available() else spec
    if spec == "auto" and not torch.cuda.is_available():
        resolved = "cpu"
    device = torch.device(resolved)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
    return device


def normalize_feature_type(feature_type: str) -> tuple[str, bool]:
    """Return the extractor feature name and whether anchor-word text is required."""
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"Unknown feature type: {feature_type}")
    if feature_type == "anchor_word":
        return "word_embeddings", True
    return feature_type, False


def feature_model_name(
    feature_type: str,
    wav2vec_model: str = "facebook/wav2vec2-large-xlsr-53",
    openai_model: str = "text-embedding-3-large",
) -> str | None:
    if feature_type in {"wav2vec2", "flipped_wav2vec2"}:
        return wav2vec_model
    if feature_type in {"word_embeddings", "sentence_embeddings"}:
        return openai_model
    return None


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Iterable[str] | None = None,
) -> argparse.Namespace:
    """Load parser defaults from ``--config`` and let CLI flags take precedence."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    preliminary, _ = pre_parser.parse_known_args(argv)

    if preliminary.config is not None:
        with preliminary.config.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        defaults = loaded.get("args", loaded)
        if not isinstance(defaults, dict):
            parser.error("The configuration file must contain a mapping or an 'args' mapping.")

        destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - destinations)
        if unknown:
            parser.error(f"Unknown configuration key(s): {', '.join(unknown)}")
        parser.set_defaults(**defaults)

    return parser.parse_args(argv)


def configure_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def move_scalers_to_device(scaler: Any, device: torch.device) -> None:
    """Move tensors stored in a serialized ``BatchScaler`` to ``device``."""
    fitted = [*scaler.meg_scalers.values(), *scaler.feature_scalers.values()]
    for item in fitted:
        if item.center_ is not None:
            item.center_ = item.center_.to(device)
        if item.scale_ is not None:
            item.scale_ = item.scale_.to(device)
        item.device = device
    scaler.device = device


def unique_directory(path: Path) -> Path:
    """Return ``path`` or the first available ``path-N`` sibling."""
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return vars(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
