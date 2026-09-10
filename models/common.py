# Building blocks for the model
# Adapted from Meta's BrainMagick project:
# https://github.com/facebookresearch/brainmagick

from functools import partial
import math
import os
import typing as tp

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MNE_USE_NUMBA", "false")

import mne
import torch
from torch import nn
from mne_bids import BIDSPath, read_raw_bids
import logging

mne.set_log_level('ERROR')  # Options: 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'

class ScaledEmbedding(nn.Module):
    """Scale up learning rate for the embedding, otherwise, it can move too slowly.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """
    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class ConvSequence(nn.Module):

    def __init__(self, channels: tp.Sequence[int], kernel: int = 4, dilation_growth: int = 1,
                 dilation_period: tp.Optional[int] = None, stride: int = 2,
                 dropout: float = 0.0, leakiness: float = 0.0, groups: int = 1,
                 decode: bool = False, batch_norm: bool = False, dropout_input: float = 0,
                 skip: bool = False, scale: tp.Optional[float] = None, rewrite: bool = False,
                 activation_on_last: bool = True, post_skip: bool = False, glu: int = 0,
                 glu_context: int = 0, glu_glu: bool = True, activation: tp.Any = None) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation *= dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x


class BIDSPositionGetter:
    INVALID = -0.1

    def __init__(self, dataset_cfg):
        """
        Compute and cache the (n_channels, 2) normalised sensor position matrix.

        Positions are fixed per MEG system, so they are computed once at init
        from the dataset config (layout name + channel picks) — no raw file I/O
        at forward time.
        """
        layout = mne.channels.read_layout(dataset_cfg.meg_layout)

        # Build clean-name → layout-index map (strip MNE suffixes like "-2622")
        layout_map = {}
        for i, lname in enumerate(layout.names):
            layout_map[lname.rsplit("-", 1)[0]] = i

        # Get channel names in the same order the data pipeline produces.
        # Read one BIDS file and apply the same picking logic as dataset.py.
        first_sub = sorted(dataset_cfg.bids_root.glob("sub-*"))[0].name.replace("sub-", "")
        bp_kwargs = dict(root=dataset_cfg.bids_root, subject=first_sub, datatype="meg")
        if dataset_cfg.name == "schoffelen2019":
            bp_kwargs["task"] = "auditory"
        else:
            bp_kwargs.update(session="0", task="0")
        raw = read_raw_bids(BIDSPath(**bp_kwargs), verbose="ERROR")

        if dataset_cfg.meg_picks is not None:
            pick_idx = mne.pick_channels_regexp(raw.info["ch_names"],
                                                regexp=dataset_cfg.meg_picks)
        else:
            pick_idx = mne.pick_types(raw.info, meg=True, ref_meg=dataset_cfg.meg_ref_meg,
                                      eeg=False, stim=False, eog=False, ecg=False)
        ch_names = [raw.info["ch_names"][i] for i in pick_idx]

        # Exclude bad channels (strip suffix before comparing, e.g. "MLC11-4304" → "MLC11")
        bad = set(dataset_cfg.meg_exclude)
        if bad:
            ch_names = [n for n in ch_names if n.rsplit('-', 1)[0] not in bad]

        # Match channels to layout positions
        indexes, valid_indexes = [], []
        for idx, name in enumerate(ch_names):
            clean = name.rsplit("-", 1)[0]
            if clean in layout_map:
                indexes.append(layout_map[clean])
                valid_indexes.append(idx)

        positions = torch.full((len(ch_names), 2), self.INVALID)
        x, y = layout.pos[indexes, :2].T
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        positions[valid_indexes, 0] = torch.tensor(x, dtype=torch.float32)
        positions[valid_indexes, 1] = torch.tensor(y, dtype=torch.float32)

        self._positions = positions  # (n_channels, 2)
        logging.info("Loaded %s sensor layout: %d/%d channels matched.",
                     dataset_cfg.meg_layout, len(valid_indexes), len(ch_names))

    def get_positions(self, meg, batch):
        B, C, T = meg.shape
        return self._positions[:C].unsqueeze(0).expand(B, -1, -1).to(meg.device)

    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)


class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """
    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2)**0.5
        assert int(n_freqs ** 2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2)**0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb

class ChannelDropout(nn.Module):
    def __init__(self, dropout: float = 0.1, rescale: bool = True, dataset_cfg=None):
        """
        Args:
            dropout: dropout radius in normalized [0, 1] coordinates.
            rescale: at valid, rescale all channels.
        """
        super().__init__()
        self.dropout = dropout
        self.rescale = rescale
        self.position_getter = BIDSPositionGetter(dataset_cfg)

    def forward(self, meg, batch):
        if not self.dropout:
            return meg

        B, C, T = meg.shape
        meg = meg.clone()
        positions = self.position_getter.get_positions(meg, batch)
        valid = (~self.position_getter.is_invalid(positions)).float()
        meg = meg * valid[:, :, None]

        if self.training:
            center_to_ban = torch.rand(2, device=meg.device)
            kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
            meg = meg * kept.float()[:, :, None]
            if self.rescale:
                proba_kept = torch.zeros(B, C, device=meg.device)
                n_tests = 100
                for _ in range(n_tests):
                    center_to_ban = torch.rand(2, device=meg.device)
                    kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
                    proba_kept += kept.float() / n_tests
                meg = meg / (1e-8 + proba_kept[:, :, None])

        return meg


class ChannelMerger(nn.Module):
    def __init__(self, chout: int, pos_dim: int = 256,
                 dropout: float = 0, usage_penalty: float = 0.,
                 n_subjects: int = 200, per_subject: bool = False,
                 dataset_cfg=None):
        super().__init__()
        assert pos_dim % 4 == 0
        self.position_getter = BIDSPositionGetter(dataset_cfg)
        self.per_subject = per_subject
        if self.per_subject:
            self.heads = nn.Parameter(torch.randn(n_subjects, chout, pos_dim, requires_grad=True))
        else:
            self.heads = nn.Parameter(torch.randn(chout, pos_dim, requires_grad=True))
        self.heads.data /= pos_dim ** 0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, meg, subjects, batch):
        B, C, T = meg.shape
        meg = meg.clone()
        positions = self.position_getter.get_positions(meg, batch)
        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=meg.device)
        score_offset[self.position_getter.is_invalid(positions)] = float('-inf')

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=meg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')

        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            heads = self.heads.gather(0, subjects.view(-1, 1, 1).expand(-1, cout, pos_dim))
        else:
            heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", meg, weights)
        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out
