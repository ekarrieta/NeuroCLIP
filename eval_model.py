"""Evaluate a NeuroCLIP checkpoint and export analysis-ready retrieval results."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from features import FeatureExtractor
from forward import BatchProcessor
from models.losses import DClipLoss
from models.meg_encoder import MEGEncoder
from norm import ScaleAndClamp
from readers.dataset import AlignConfig, AudioConfig, DatasetConfig, MEGConfig, SegmentsDataset
from utils import (
    FEATURE_TYPES,
    PROJECT_ROOT,
    configure_logging,
    feature_model_name,
    move_scalers_to_device,
    normalize_feature_type,
    parse_args_with_config,
    resolve_device,
    save_json,
    set_seed,
    unique_directory,
)


def _item(values: Any, index: int, default: Any = "") -> Any:
    if values is None:
        return default
    try:
        value = values[index]
    except (IndexError, KeyError, TypeError):
        return default
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Evaluator:
    """Compute retrieval metrics and one analysis row per query segment."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        dataloader: DataLoader,
        batch_processor: BatchProcessor,
        loss: DClipLoss,
        checkpoint_dir: Path,
        negatives: int | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.batch_processor = batch_processor
        self.loss = loss
        self.checkpoint_dir = Path(checkpoint_dir)
        self.negatives = negatives

    def load_best_model_and_evaluate(self) -> tuple[dict[str, float | int], pd.DataFrame]:
        checkpoint_path = self.checkpoint_dir / "best_model.pth"
        logging.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return self.evaluate()

    @torch.inference_mode()
    def evaluate(self) -> tuple[dict[str, float | int], pd.DataFrame]:
        self.model.eval()
        correct_at_1 = 0
        correct_at_10 = 0
        reciprocal_rank_sum = 0.0
        loss_sum = 0.0
        loss_weight = 0
        candidate_sum = 0
        rows: list[dict[str, Any]] = []

        for batch_number, batch in enumerate(
            tqdm(self.dataloader, desc="Evaluate", ncols=100)
        ):
            estimate, targets = self.batch_processor.process_batch(batch)
            batch_size = targets.shape[0]
            if batch_size <= 1:
                logging.warning(
                    "Skipping evaluation batch %d because it has fewer than two items.",
                    batch_number,
                )
                continue

            stim_ids = [str(value) for value in batch["stim_id"]]
            batch_loss = self.loss(estimate, targets, stim_ids)
            loss_sum += float(batch_loss.item()) * batch_size
            loss_weight += batch_size

            for query_index in range(batch_size):
                other_indices = torch.arange(batch_size, device=self.device)
                other_indices = other_indices[other_indices != query_index]
                if self.negatives is not None and self.negatives < other_indices.numel():
                    order = torch.randperm(other_indices.numel(), device=self.device)
                    other_indices = other_indices[order[: self.negatives]]

                candidate_indices = torch.cat(
                    [torch.tensor([query_index], device=self.device), other_indices]
                )
                candidates = targets[candidate_indices]
                scores = self.loss.get_scores(
                    estimate[query_index : query_index + 1], candidates
                )[0]
                ranking = torch.argsort(scores, descending=True)

                candidate_ids = [stim_ids[index] for index in candidate_indices.tolist()]
                positive_mask = torch.tensor(
                    [
                        candidate_id == stim_ids[query_index]
                        for candidate_id in candidate_ids
                    ],
                    dtype=torch.bool,
                    device=self.device,
                )
                ranked_positive = positive_mask[ranking]
                rank = int(torch.nonzero(ranked_positive, as_tuple=False)[0].item()) + 1
                top1 = rank == 1
                top10 = rank <= min(10, len(candidate_indices))
                retrieved_index = int(candidate_indices[ranking[0]].item())

                positive_score = float(scores[positive_mask].max().item())
                negative_scores = scores[~positive_mask]
                best_negative_score = (
                    float(negative_scores.max().item())
                    if negative_scores.numel()
                    else float("nan")
                )

                correct_at_1 += int(top1)
                correct_at_10 += int(top10)
                reciprocal_rank_sum += 1.0 / rank
                candidate_sum += len(candidate_indices)

                rows.append(
                    {
                        "query_number": len(rows),
                        "batch_number": batch_number,
                        "query_batch_index": query_index,
                        "segment_uid": _item(batch.get("segment_uid"), query_index),
                        "stim_id": stim_ids[query_index],
                        "subject": _item(batch.get("subject"), query_index),
                        "session": _item(batch.get("session"), query_index),
                        "task": _item(batch.get("task"), query_index),
                        "condition": _item(batch.get("condition"), query_index),
                        "anchor_word": _item(batch.get("anchor_word"), query_index),
                        "words": _item(batch.get("words"), query_index),
                        "retrieved_segment_uid": _item(
                            batch.get("segment_uid"), retrieved_index
                        ),
                        "retrieved_stim_id": stim_ids[retrieved_index],
                        "retrieved_subject": _item(
                            batch.get("subject"), retrieved_index
                        ),
                        "rank": rank,
                        "top1": top1,
                        "top10": top10,
                        "reciprocal_rank": 1.0 / rank,
                        "candidate_count": len(candidate_indices),
                        "positive_score": positive_score,
                        "retrieved_score": float(scores[ranking[0]].item()),
                        "best_negative_score": best_negative_score,
                        "score_margin": positive_score - best_negative_score,
                    }
                )

        total = len(rows)
        if total == 0:
            raise RuntimeError(
                "Evaluation produced no queries. Use a split and batch size with at least two items."
            )

        metrics: dict[str, float | int] = {
            "top1_percent": 100.0 * correct_at_1 / total,
            "top10_percent": 100.0 * correct_at_10 / total,
            "mrr": reciprocal_rank_sum / total,
            "loss": loss_sum / loss_weight,
            "n_queries": total,
            "mean_candidates": candidate_sum / total,
        }
        logging.info(
            "Top-1 %.2f%% | Top-10 %.2f%% | MRR %.4f | loss %.4f",
            metrics["top1_percent"],
            metrics["top10_percent"],
            metrics["mrr"],
            metrics["loss"],
        )
        return metrics, pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a NeuroCLIP checkpoint and export per-query retrieval data."
    )
    parser.add_argument("--config", type=Path, help="YAML file providing argument defaults.")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", type=Path)
    parser.add_argument(
        "--dataset", choices=("gwilliams2022", "schoffelen2019")
    )
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument(
        "--segments",
        "--data",
        type=Path,
        help="Segments TSV; overrides the default cache path.",
    )
    parser.add_argument("--bids-root", type=Path)
    parser.add_argument("--stim-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reuse the output directory instead of adding a suffix.",
    )
    parser.add_argument(
        "--split", choices=("train", "valid", "test"), default="test"
    )
    parser.add_argument("--length", type=float)
    parser.add_argument(
        "--feature-type", "--feature_type", choices=FEATURE_TYPES
    )
    parser.add_argument(
        "--anchor-word", "--anchor_word", action="store_true", default=None
    )
    parser.add_argument("--model", choices=("cnn", "conformer"))
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "models" / "meg_encoder.yaml",
    )
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--num-layers", "--nlayers", dest="num_layers", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--num-heads", "--nheads", dest="num_heads", type=int)
    parser.add_argument(
        "--initial-linear", "--initial_linear", dest="initial_linear", type=int
    )
    parser.add_argument("--meg-hidden", "--meg_hidden", dest="meg_hidden", type=int)
    parser.add_argument("--clamp", type=float)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=1000
    )
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers", type=int, default=3
    )
    parser.add_argument(
        "--negatives",
        "--wer-negatives",
        "--wer_negatives",
        dest="negatives",
        type=int,
    )
    parser.add_argument("--seed", type=int, default=2036)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, for example auto, cpu, cuda, or cuda:1.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle segments before forming candidate batches.",
    )
    parser.add_argument("--wav2vec-model")
    parser.add_argument("--openai-model")
    return parser


def _saved_value(
    cli_value: Any, saved: dict[str, Any], key: str, default: Any
) -> Any:
    return cli_value if cli_value is not None else saved.get(key, default)


def main(argv: list[str] | None = None) -> dict[str, float | int]:
    parser = build_parser()
    args = parse_args_with_config(parser, argv)
    if args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required (or set checkpoint_dir in --config)")
    if args.negatives is not None and args.negatives < 1:
        parser.error("--negatives must be positive")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")

    configure_logging()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_path = checkpoint_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_args = checkpoint.get("args", {})
    dataset_name = _saved_value(
        args.dataset, saved_args, "dataset", "gwilliams2022"
    )
    dataset_config_path = (
        args.dataset_config
        or PROJECT_ROOT / "readers" / f"{dataset_name}.yaml"
    )
    dataset_cfg = DatasetConfig.from_yaml(dataset_config_path)
    if args.bids_root is not None:
        dataset_cfg.bids_root = args.bids_root.resolve()
    if args.stim_root is not None:
        dataset_cfg.stim_root = args.stim_root.resolve()

    cache_setting = _saved_value(args.cache_dir, saved_args, "cache_dir", None)
    cache_dir = (
        Path(cache_setting)
        if cache_setting
        else PROJECT_ROOT / "cache" / dataset_cfg.cache_subdir
    ).resolve()
    length = float(_saved_value(args.length, saved_args, "length", 3.0))
    requested_feature = _saved_value(
        args.feature_type, saved_args, "feature_type", "wav2vec2"
    )
    feature_type, inferred_anchor = normalize_feature_type(requested_feature)
    anchor_word = inferred_anchor or bool(
        _saved_value(args.anchor_word, saved_args, "anchor_word", False)
    )
    model_name = _saved_value(args.model, saved_args, "model", "cnn")
    wav2vec_model = _saved_value(
        args.wav2vec_model, saved_args, "wav2vec_model",
        "facebook/wav2vec2-large-xlsr-53",
    )
    openai_model = _saved_value(
        args.openai_model, saved_args, "openai_model",
        "text-embedding-3-large",
    )
    clamp = float(_saved_value(args.clamp, saved_args, "clamp", 20.0))

    model_config = copy.deepcopy(checkpoint.get("model_config"))
    if model_config is None:
        with args.model_config.open(encoding="utf-8") as handle:
            model_config = yaml.safe_load(handle)
    model_config["in_channels"]["meg"] = dataset_cfg.n_channels
    model_config["n_subjects"] = dataset_cfg.n_subjects
    initial_linear = _saved_value(
        args.initial_linear, saved_args, "initial_linear", None
    )
    meg_hidden = _saved_value(args.meg_hidden, saved_args, "meg_hidden", None)
    if initial_linear is not None:
        model_config["initial_linear"] = initial_linear
    if meg_hidden is not None:
        model_config["hidden"]["meg"] = meg_hidden

    num_layers = _saved_value(
        args.num_layers,
        saved_args,
        "num_layers",
        saved_args.get("nlayers", 6),
    )
    num_heads = _saved_value(
        args.num_heads,
        saved_args,
        "num_heads",
        saved_args.get("nheads", 8),
    )
    model = MEGEncoder(
        **model_config,
        meg_encoder=model_name,
        tf_dropout=float(
            _saved_value(args.dropout, saved_args, "dropout", 0.2)
        ),
        num_layers=int(num_layers),
        tf_hidden=int(_saved_value(args.hidden, saved_args, "hidden", 320)),
        nheads=int(num_heads),
        dataset_cfg=dataset_cfg,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    segments_setting = _saved_value(args.segments, saved_args, "segments", None)
    segments_path = (
        Path(segments_setting)
        if segments_setting
        else cache_dir / "annotations" / f"segments_{length:.1f}s.tsv"
    )
    if not segments_path.is_file():
        raise FileNotFoundError(f"Segments table not found: {segments_path}")
    segments = pd.read_csv(segments_path, sep="\t")
    dataset = SegmentsDataset(
        segments_df=segments,
        split=args.split,
        feature_type=feature_type,
        meg_cfg=MEGConfig(
            dataset_cfg.bids_root,
            dataset_cfg.meg_picks,
            dataset_cfg.meg_target_hz,
            cache_dir / "meg",
        ),
        audio_cfg=AudioConfig(
            dataset_cfg.stim_root,
            dataset_cfg.audio_resample_hz,
            cache_dir / "audio",
        ),
        align_cfg=AlignConfig(),
        dataset_cfg=dataset_cfg,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    scaler_path = cache_dir / "scalers" / "batch_scalers.pkl"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Training scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    move_scalers_to_device(scaler, device)
    normalizer = ScaleAndClamp(scaler, limit=clamp)

    extractor = FeatureExtractor(
        feature_type=feature_type,
        model_name=feature_model_name(
            feature_type, wav2vec_model, openai_model
        ),
        device=device,
        feature_dim=model_config["out_channels"],
        segment_length=length,
        cache_dir=cache_dir,
    )
    processor = BatchProcessor(
        model,
        device,
        normalizer,
        feature_type,
        anchor_word,
        extractor,
        model_config,
    )
    evaluator = Evaluator(
        model,
        device,
        dataloader,
        processor,
        DClipLoss().to(device),
        checkpoint_dir,
        args.negatives,
    )
    metrics, queries = evaluator.evaluate()

    base_output = (
        args.output_dir or checkpoint_dir / "evaluation" / args.split
    )
    output_dir = base_output if args.overwrite else unique_directory(base_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": checkpoint_dir.name,
        "dataset": dataset_name,
        "split": args.split,
        "segments": str(segments_path),
        "feature_type": requested_feature,
        "model": model_name,
        "batch_size": args.batch_size,
        "negatives": args.negatives,
        "seed": args.seed,
        "metrics": metrics,
    }
    save_json(report, output_dir / "metrics.json")
    queries.to_csv(output_dir / "queries.tsv", sep="\t", index=False)
    logging.info(
        "Saved metrics and %d query rows to %s", len(queries), output_dir
    )
    return metrics


if __name__ == "__main__":
    main()
