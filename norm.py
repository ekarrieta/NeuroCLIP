# Normalization and scaling utilities for MEG data

import torch
from collections import defaultdict
from typing import Dict
import logging
from tqdm import tqdm

class RobustScaler:
    """
    RobustScaler for MEG data. Fits on the training data and scales data using
    specified quantiles (e.g., 25th and 75th percentiles).
    """
    def __init__(self, lowq=0.25, highq=0.75, subsample=1.0, device="cpu"):
        self.lowq = lowq
        self.highq = highq
        self.subsample = subsample
        self.device = device
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        samples, dimension = X.shape
        self.center_ = torch.empty(dimension, device=self.device)
        self.scale_ = torch.empty(dimension, device=self.device)

        for d in range(dimension):
            col = X[:, d]
            keep = torch.rand_like(col) < self.subsample
            col = col[keep]
            col, _ = col.sort()
            low, med, high = [
                col[int(q * len(col))].item() for q in [self.lowq, 0.5, self.highq]
            ]
            self.center_[d] = med
            self.scale_[d] = high - low
            if self.scale_[d] == 0:
                self.scale_[d] = 1.0  # Avoid division by zero

        return self

    def transform(self, X):
        return (X - self.center_) / self.scale_


class StandardScaler:
    """
    StandardScaler for features. Computes the mean and standard deviation for
    normalization and scales data accordingly.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        self.center_ = X.mean(dim=0).to(self.device)
        self.scale_ = X.std(dim=0).to(self.device)
        self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
        return self

    def transform(self, X):
        return (X - self.center_) / self.scale_

class TimeSeriesScaler:
    """
    Scaler for 1D time-series features (envelope, pitch, ...).
    Computes global mean and std across all samples and timepoints.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Args:
            X: [B*T] flattened time-series data
        """
        self.center_ = X.mean().to(self.device)
        self.scale_ = X.std().to(self.device)
        if self.scale_ == 0:
            self.scale_ = torch.tensor(1.0, device=self.device)
        return self

    def transform(self, X):
        """
        Args:
            X: [B, T] or [B, 1024, T] time-series batch
        """
        return (X - self.center_) / self.scale_


class BatchScaler:
    """
    Handles normalization of MEG and feature data in batches. Fits scalers for
    MEG and features, and applies normalization during training.
    """
    def __init__(self, device="cuda", n_samples_per_recording=200, n_samples_features=8000):
        self.device = device
        self.n_samples_per_recording = n_samples_per_recording
        self.n_samples_features = n_samples_features
        self.meg_scalers: Dict[int, RobustScaler] = {}  # One per subject/recording
        self.feature_scalers: Dict[str, object] = {} # One per feature type

    def fit(self, dataloader, feature_extractor, feature_type, sr, target_len):
        # If meg_scalers already exist, we reuse them and skip MEG fitting
        meg_already_fitted = len(self.meg_scalers) > 0
        if meg_already_fitted:
            logging.info(
                f"BatchScaler.fit: found existing MEG scalers "
                f"({len(self.meg_scalers)} recordings). Skipping MEG fitting."
            )
            meg_data_count = None
            meg_data_samples = None
        else:
            meg_data_count = defaultdict(int)
            meg_data_samples = defaultdict(list)
        feature_data_samples = []

        logging.info(f'Fitting scalers for feature type: {feature_type}')

        # Collect samples
        for batch in tqdm(dataloader, mininterval=2.0, miniters=25):
            # Collect MEG samples per recording
            if not meg_already_fitted:
                for meg, recording_idx in zip(batch['meg'], batch['recording_index']):
                    recording_idx = recording_idx.item()
                    if meg_data_count[recording_idx] < self.n_samples_per_recording:
                        meg_data_samples[recording_idx].append(meg)
                        meg_data_count[recording_idx] += 1

            # Collect feature samples (only for feature types that need normalization)
            if feature_type not in ['word_embeddings', 'sentence_embeddings', 'static_embeddings', 'vad', 'noise']:
                if len(feature_data_samples) < self.n_samples_features:
                    # Extract features from audio (one at a time to save memory)
                    for i in range(min(batch['features'].shape[0], self.n_samples_features - len(feature_data_samples))):
                        with torch.no_grad():
                            single_audio = batch['features'][i:i+1]  # [1, seq_len]
                            features = feature_extractor.extract_features(
                                single_audio,
                                sr=sr,
                                target_len=target_len,
                                segment_ids=batch['segment_uid'][i:i+1]
                            )
                            feature_data_samples.append(features[0].cpu())
                            # Clear GPU cache every 100 samples
                            if len(feature_data_samples) % 100 == 0:
                                torch.cuda.empty_cache()

            # Check if we have enough samples
            enough_meg = True
            if not meg_already_fitted:
                enough_meg = all(count >= self.n_samples_per_recording for count in meg_data_count.values())

            enough_feat = (len(feature_data_samples) >= self.n_samples_features or feature_type in ['word_embeddings', 'sentence_embeddings', 'static_embeddings', 'vad', 'noise'])

            if enough_meg and enough_feat:
                break

        # Fit MEG scalers
        if not meg_already_fitted:
            logging.info('Fitting MEG scalers...')
            for recording_index, recordings in meg_data_samples.items():
                recordings_concat = torch.stack(recordings).to(self.device)
                scaler = RobustScaler(device=self.device)
                scaler.fit(flatten(recordings_concat))
                self.meg_scalers[recording_index] = scaler

        # Fit feature scalers based on feature type
        if feature_type in ['wav2vec2', 'flipped_wav2vec2', 'mfcc', 'mel_spectrogram']:
            logging.info(f'Fitting StandardScaler for {feature_type} features...')
            features_concat = torch.stack(feature_data_samples).to(self.device)  # [N, 1024, T]
            scaler = StandardScaler(device=self.device)
            scaler.fit(flatten(features_concat))  # [N*T, 1024]
            self.feature_scalers[feature_type] = scaler

        elif feature_type in ['word_embeddings', 'sentence_embeddings', 'static_embeddings', 'vad', 'noise']:
            logging.info(f"{feature_type} features do not require normalization")

        else:
            logging.info(f'Fitting TimeSeriesScaler for {feature_type} features...')
            features_concat = torch.stack(feature_data_samples).to(self.device)  # [N, T]
            scaler = TimeSeriesScaler(device=self.device)
            # Flatten all samples and timepoints into single vector
            scaler.fit(features_concat.reshape(-1))  # [N*T]
            self.feature_scalers[feature_type] = scaler

    def transform(self, batch, features, feature_type, only_meg):
        # Normalize MEG data
        for i, (meg, recording_index) in enumerate(zip(batch["meg"], batch["recording_index"])):
            scaler = self.meg_scalers[recording_index.item()]
            batch["meg"][i] = (scaler.transform(meg.T)).T

        if only_meg:
            return batch

        if feature_type in ['wav2vec2', 'flipped_wav2vec2', 'mfcc', 'mel_spectrogram']:
            assert features.dim() == 3
            # Normalize feature data using StandardScaler (per feature dimension)
            scaler = self.feature_scalers[feature_type]
            features = unflatten(
                scaler.transform(flatten(features)),
                features.shape,
            )
            return batch, features

        elif feature_type in ['word_embeddings', 'sentence_embeddings', 'static_embeddings', 'vad', 'noise']:
            # Should be covered by only_meg=True but just in case
            return batch

        else:
            # Normalize using TimeSeriesScaler (global mean/std)
            scaler = self.feature_scalers[feature_type]
            features = scaler.transform(features)
            return batch, features

def flatten(x):
    """Converts`x` from [B, C, T] to [B * T, C].
    """
    if x.ndimension() == 3:
        return x.permute(0, 2, 1).reshape(-1, x.shape[1])
    elif x.ndimension() == 2:
        return x.permute(1, 0)
    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions.")


def unflatten(x, shape):
    """Converts from `[B * prod(shape), C]` to `[B, C, *shape]`.
    """
    return x.view(shape[0], shape[2], -1).permute(0, 2, 1).contiguous()


class ScaleAndClamp:
    """
    Rescales the input MEG and features and clips outliers.
    """

    def __init__(self, scaler: BatchScaler, limit):
        self.scaler = scaler
        self.limit = limit

    def __call__(self, batch, features, feature_type, only_meg):
        """
        Applies normalization and clips outliers.

        Args:
            batch: Dictionary containing batch data.

        Returns:
            Normalize batch.
        """
        if only_meg:
            batch = self.scaler.transform(batch, features, feature_type, only_meg)
            batch['meg'].clamp_(-self.limit, self.limit)
            return batch
        else:
            batch, features = self.scaler.transform(batch, features, feature_type, only_meg)
            batch['meg'].clamp_(-self.limit, self.limit)
            return batch, features
