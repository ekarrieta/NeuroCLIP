"""Batch preprocessing and model-forward orchestration."""

from __future__ import annotations

from typing import Any

import torch


FEATURES_WITHOUT_SCALING = {
    "word_embeddings",
    "sentence_embeddings",
    "static_embeddings",
    "vad",
    "noise",
}


class BatchProcessor:
    """Turn a collated dataset batch into aligned model and target tensors."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        normalizer: Any,
        feature_type: str,
        anchor_word: bool,
        feature_extractor: Any,
        model_config: dict[str, Any],
    ) -> None:
        self.model = model
        self.device = device
        self.normalizer = normalizer
        self.feature_type = feature_type
        self.anchor_word = anchor_word
        self.feature_extractor = feature_extractor
        self.out_channels = model_config["out_channels"]

    def process_batch(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        meg = batch["meg"].to(self.device, non_blocking=True)
        if self.feature_type == "flipped_wav2vec2":
            meg = meg.flip(-1)
        batch["meg"] = meg

        subjects = batch["subject"].to(self.device, non_blocking=True)
        words = batch["anchor_word"] if self.anchor_word else batch["words"]
        features = self.feature_extractor.extract_features(
            batch["features"],
            sr=batch["wav_sr"],
            target_len=meg.shape[-1],
            words=words,
            segment_ids=batch["segment_uid"],
        ).to(self.device, non_blocking=True)

        only_meg = self.feature_type in FEATURES_WITHOUT_SCALING
        normalized = self.normalizer(
            batch, features, self.feature_type, only_meg=only_meg
        )
        if only_meg:
            batch = normalized
        else:
            batch, features = normalized

        if features.ndim == 2:
            features = features.unsqueeze(1).expand(
                -1, self.out_channels, -1
            )

        batch["features"] = features
        estimate = self.model({"meg": batch["meg"]}, subjects, batch)
        return estimate, features
