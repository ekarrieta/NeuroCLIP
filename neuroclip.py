"""Train NeuroCLIP models for MEG-to-stimulus retrieval."""

from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_model import Evaluator
from features import FeatureExtractor
from forward import BatchProcessor
from models.losses import DClipLoss
from models.meg_encoder import MEGEncoder
from norm import BatchScaler, ScaleAndClamp
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
)


NO_FEATURE_SCALING = {
    "word_embeddings",
    "sentence_embeddings",
    "static_embeddings",
    "vad",
    "noise",
}


class NeuroCLIPTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        self.dataset_cfg = DatasetConfig.from_yaml(args.dataset_config)
        if args.bids_root is not None:
            self.dataset_cfg.bids_root = args.bids_root.resolve()
        if args.stim_root is not None:
            self.dataset_cfg.stim_root = args.stim_root.resolve()

        self.cache_dir = (
            args.cache_dir
            or PROJECT_ROOT / "cache" / self.dataset_cfg.cache_subdir
        ).resolve()
        self.run_dir = (
            args.output_dir / self.dataset_cfg.cache_subdir / args.name
        ).resolve()
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory is not empty: {self.run_dir}. Choose a new --name."
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(self.run_dir / "train.log")
        logging.info("Using %s", self.device)

        self.requested_feature_type = args.feature_type
        self.feature_type, self.anchor_word = normalize_feature_type(
            args.feature_type
        )
        with args.model_config.open(encoding="utf-8") as handle:
            self.model_config = copy.deepcopy(yaml.safe_load(handle))
        self.model_config["in_channels"]["meg"] = self.dataset_cfg.n_channels
        self.model_config["n_subjects"] = self.dataset_cfg.n_subjects
        if args.initial_linear is not None:
            self.model_config["initial_linear"] = args.initial_linear
        if args.meg_hidden is not None:
            self.model_config["hidden"]["meg"] = args.meg_hidden

        model_kwargs = copy.deepcopy(self.model_config)
        self.model = MEGEncoder(
            **model_kwargs,
            meg_encoder=args.model,
            tf_dropout=args.dropout,
            num_layers=args.num_layers,
            tf_hidden=args.hidden,
            nheads=args.num_heads,
            dataset_cfg=self.dataset_cfg,
        ).to(self.device)
        self.loss = DClipLoss().to(self.device)
        self._init_data()

        self.feature_extractor = FeatureExtractor(
            feature_type=self.feature_type,
            model_name=feature_model_name(
                self.feature_type, args.wav2vec_model, args.openai_model
            ),
            device=self.device,
            feature_dim=self.model_config["out_channels"],
            segment_length=args.length,
            cache_dir=self.cache_dir,
        )
        self._init_scaler()
        self.batch_processor = BatchProcessor(
            self.model,
            self.device,
            self.normalizer,
            self.feature_type,
            self.anchor_word,
            self.feature_extractor,
            self.model_config,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        self.history: list[dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self._save_run_config()
        logging.info(
            "Model has %d trainable parameters",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )

    def _init_data(self) -> None:
        segments_path = (
            self.args.segments
            or self.cache_dir
            / "annotations"
            / f"segments_{self.args.length:.1f}s.tsv"
        )
        if not segments_path.is_file():
            raise FileNotFoundError(
                f"Segments table not found: {segments_path}. "
                "Run dataset preprocessing first."
            )
        self.segments_path = segments_path.resolve()
        segments = pd.read_csv(segments_path, sep="\t")
        meg_config = MEGConfig(
            self.dataset_cfg.bids_root,
            self.dataset_cfg.meg_picks,
            self.dataset_cfg.meg_target_hz,
            self.cache_dir / "meg",
        )
        audio_config = AudioConfig(
            self.dataset_cfg.stim_root,
            self.dataset_cfg.audio_resample_hz,
            self.cache_dir / "audio",
        )
        common = {
            "segments_df": segments,
            "feature_type": self.feature_type,
            "meg_cfg": meg_config,
            "audio_cfg": audio_config,
            "align_cfg": AlignConfig(),
            "dataset_cfg": self.dataset_cfg,
        }
        self.train_dataset = SegmentsDataset(split="train", **common)
        self.valid_dataset = SegmentsDataset(split="valid", **common)
        self.test_dataset = SegmentsDataset(split="test", **common)
        if len(self.train_dataset) < 2 or len(self.valid_dataset) < 2:
            raise ValueError(
                "Training and validation splits must each contain at least two segments."
            )
        if self.args.final_eval and len(self.test_dataset) < 2:
            raise ValueError(
                "The test split must contain at least two segments for final evaluation."
            )

        loader_options = {
            "num_workers": self.args.num_workers,
            "pin_memory": self.args.pin_memory,
            "persistent_workers": self.args.num_workers > 0,
        }
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=len(self.train_dataset) >= self.args.batch_size,
            **loader_options,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )
        logging.info(
            "Loaded train/valid/test splits with %d/%d/%d segments",
            len(self.train_dataset),
            len(self.valid_dataset),
            len(self.test_dataset),
        )

    def _init_scaler(self) -> None:
        scaler_path = self.cache_dir / "scalers" / "batch_scalers.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        if scaler_path.is_file():
            logging.info("Loading scalers from %s", scaler_path)
            self.scaler = joblib.load(scaler_path)
            move_scalers_to_device(self.scaler, self.device)
        else:
            self.scaler = BatchScaler(
                n_samples_per_recording=self.args.scaler_meg_samples,
                n_samples_features=self.args.scaler_feature_samples,
                device=self.device,
            )

        feature_needs_scaler = self.feature_type not in NO_FEATURE_SCALING
        needs_fit = not self.scaler.meg_scalers or (
            feature_needs_scaler
            and self.feature_type not in self.scaler.feature_scalers
        )
        if needs_fit:
            scaler_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.scaler_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=self.args.pin_memory,
                drop_last=False,
            )
            sample = next(iter(scaler_loader))
            self.scaler.fit(
                scaler_loader,
                self.feature_extractor,
                self.feature_type,
                sample["wav_sr"],
                sample["meg"].shape[-1],
            )
            joblib.dump(self.scaler, scaler_path)
            logging.info("Saved fitted scalers to %s", scaler_path)
        self.normalizer = ScaleAndClamp(self.scaler, limit=self.args.clamp)

    def _save_run_config(self) -> None:
        arguments = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self.args).items()
        }
        payload = {
            "args": arguments,
            "resolved": {
                "bids_root": str(self.dataset_cfg.bids_root),
                "stim_root": str(self.dataset_cfg.stim_root),
                "cache_dir": str(self.cache_dir),
                "segments": str(self.segments_path),
                "run_dir": str(self.run_dir),
            },
            "model_config": self.model_config,
        }
        with (self.run_dir / "run_config.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    def _run_train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for batch in tqdm(
            self.train_loader, desc="Train", ncols=100, leave=False
        ):
            estimate, targets = self.batch_processor.process_batch(batch)
            loss = self.loss(estimate, targets, batch["stim_id"])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
        return total / len(self.train_loader)

    @torch.inference_mode()
    def _run_valid_epoch(self) -> float:
        self.model.eval()
        total = 0.0
        for batch in tqdm(
            self.valid_loader, desc="Valid", ncols=100, leave=False
        ):
            estimate, targets = self.batch_processor.process_batch(batch)
            total += float(
                self.loss(estimate, targets, batch["stim_id"]).item()
            )
        return total / len(self.valid_loader)

    def _save_checkpoint(self, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_val_loss,
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(self.args).items()
                },
                "model_config": self.model_config,
            },
            self.run_dir / "best_model.pth",
        )

    def _save_history(self) -> None:
        pd.DataFrame(self.history).to_csv(
            self.run_dir / "history.csv", index=False
        )

    def _save_embedding_cache(self) -> None:
        if self.feature_type == "word_embeddings":
            self.feature_extractor.save_word_cache(
                self.cache_dir / "embeddings" / "word_cache.pkl"
            )
        elif self.feature_type == "sentence_embeddings":
            self.feature_extractor.save_sentence_cache(
                self.cache_dir / "embeddings" / "sentence_cache.pkl"
            )

    def train(self) -> dict[str, Any]:
        started = time.time()
        epochs_completed = 0
        for epoch in range(self.args.epochs):
            train_loss = self._run_train_epoch()
            valid_loss = self._run_valid_epoch()
            epochs_completed = epoch + 1
            improved = valid_loss < self.best_val_loss
            if improved:
                self.best_val_loss = valid_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch)

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "best_valid_loss": self.best_val_loss,
                    "is_best": improved,
                }
            )
            self._save_history()
            logging.info(
                "Epoch %d/%d | train %.4f | valid %.4f%s",
                epoch + 1,
                self.args.epochs,
                train_loss,
                valid_loss,
                " | checkpoint saved" if improved else "",
            )

            if (
                not improved
                and epoch - self.best_epoch >= self.args.early_stop
            ):
                logging.info(
                    "Early stopping after %d epochs without improvement",
                    self.args.early_stop,
                )
                break
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        test_metrics: dict[str, float | int] = {}
        if self.args.final_eval:
            evaluator = Evaluator(
                self.model,
                self.device,
                self.test_loader,
                self.batch_processor,
                self.loss,
                self.run_dir,
                self.args.eval_negatives,
            )
            test_metrics, queries = (
                evaluator.load_best_model_and_evaluate()
            )
            queries.to_csv(
                self.run_dir / "test_queries.tsv", sep="\t", index=False
            )

        self._save_embedding_cache()
        summary = {
            "run_name": self.args.name,
            "dataset": self.dataset_cfg.name,
            "model": self.args.model,
            "feature_type": self.requested_feature_type,
            "train_segments": len(self.train_dataset),
            "valid_segments": len(self.valid_dataset),
            "test_segments": len(self.test_dataset),
            "best_valid_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epochs_completed": epochs_completed,
            "training_seconds": time.time() - started,
            "test": test_metrics,
        }
        save_json(summary, self.run_dir / "metrics.json")
        logging.info("Run outputs saved to %s", self.run_dir)
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a NeuroCLIP retrieval model."
    )
    parser.add_argument(
        "--config", type=Path, help="YAML file providing argument defaults."
    )
    parser.add_argument(
        "--name", "-n", help="Run name; defaults to a timestamp."
    )
    parser.add_argument(
        "--dataset",
        choices=("gwilliams2022", "schoffelen2019"),
        default="gwilliams2022",
    )
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--bids-root", type=Path)
    parser.add_argument("--stim-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "models" / "meg_encoder.yaml",
    )
    parser.add_argument("--length", "-l", type=float, default=3.0)
    parser.add_argument(
        "--model", choices=("cnn", "conformer"), default="cnn"
    )
    parser.add_argument(
        "--feature-type",
        "--feature_type",
        choices=FEATURE_TYPES,
        default="wav2vec2",
    )
    parser.add_argument("--epochs", "--epoch", type=int, default=20)
    parser.add_argument(
        "--early-stop", "--earlystop", type=int, default=3
    )
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        "-lr",
        dest="learning_rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        "--decay",
        dest="weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--eval-batch-size",
        "--test_batch_size",
        dest="eval_batch_size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=3,
    )
    parser.add_argument("--seed", type=int, default=2036)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--num-layers", "--nlayers", dest="num_layers", type=int, default=6
    )
    parser.add_argument("--hidden", type=int, default=320)
    parser.add_argument(
        "--num-heads", "--nheads", dest="num_heads", type=int, default=8
    )
    parser.add_argument(
        "--initial-linear",
        "--initial_linear",
        dest="initial_linear",
        type=int,
    )
    parser.add_argument(
        "--meg-hidden", "--meg_hidden", dest="meg_hidden", type=int
    )
    parser.add_argument("--clamp", type=float, default=20.0)
    parser.add_argument(
        "--scaler-meg-samples", type=int, default=200
    )
    parser.add_argument(
        "--scaler-feature-samples", type=int, default=8000
    )
    parser.add_argument("--scaler-batch-size", type=int, default=16)
    parser.add_argument("--eval-negatives", type=int)
    parser.add_argument(
        "--final-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, for example auto, cpu, cuda, or cuda:1.",
    )
    parser.add_argument(
        "--wav2vec-model", default="facebook/wav2vec2-large-xlsr-53"
    )
    parser.add_argument(
        "--openai-model", default="text-embedding-3-large"
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parse_args_with_config(parser, argv)
    if args.dataset_config is None:
        args.dataset_config = (
            PROJECT_ROOT / "readers" / f"{args.dataset}.yaml"
        )
    if args.name is None:
        args.name = time.strftime("%Y%m%d-%H%M%S")
    if args.pin_memory is None:
        args.pin_memory = resolve_device(args.device).type == "cuda"
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size at least 2")
    if args.eval_batch_size < 2:
        parser.error("--eval-batch-size must be at least 2")
    if args.early_stop < 1:
        parser.error("--early-stop must be at least 1")
    if args.eval_negatives is not None and args.eval_negatives < 1:
        parser.error("--eval-negatives must be positive")

    mp.set_start_method("spawn", force=True)
    set_seed(args.seed)
    trainer = NeuroCLIPTrainer(args)
    return trainer.train()


if __name__ == "__main__":
    main()
