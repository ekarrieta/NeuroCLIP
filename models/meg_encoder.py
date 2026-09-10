from functools import partial
import random
import typing as tp

import torch
from torch import nn

from .common import ConvSequence, ScaledEmbedding, SubjectLayers, ChannelMerger, ChannelDropout
from .conformers import ConformerEncoder


class MEGEncoder(nn.Module):
    def __init__(self,
                 # Channels
                 in_channels: tp.Dict[str, int],
                 out_channels: int,
                 hidden: tp.Dict[str, int],
                 # Overall structure
                 depth: int = 4,
                 concatenate: bool = False,  # concatenate the inputs
                 linear_out: bool = False,
                 complex_out: bool = False,
                 meg_encoder: str = 'cnn',
                 # Transformer
                 num_layers: int = 4,
                 tf_dropout: float = 0.0,
                 tf_hidden: int = 320,
                 nheads: int = 4,
                 # Conv layer
                 kernel_size: int = 5,
                 growth: float = 1.,
                 dilation_growth: int = 2,
                 dilation_period: tp.Optional[int] = None,
                 skip: bool = False,
                 post_skip: bool = False,
                 scale: tp.Optional[float] = None,
                 rewrite: bool = False,
                 groups: int = 1,
                 glu: int = 0,
                 glu_context: int = 0,
                 glu_glu: bool = True,
                 gelu: bool = False,
                 # Dropouts, BN, activations
                 conv_dropout: float = 0.0,
                 dropout_input: float = 0.0,
                 batch_norm: bool = False,
                 relu_leakiness: float = 0.0,
                 # Subject specific settings
                 n_subjects: int = 200,
                 subject_dim: int = 64,
                 subject_layers: bool = False,
                 subject_layers_dim: str = "input",  # or hidden
                 subject_layers_id: bool = False,
                 embedding_scale: float = 1.0,
                 # Attention multi-dataset support
                 merger: bool = False,
                 merger_pos_dim: int = 256,
                 merger_channels: int = 270,
                 merger_dropout: float = 0.2,
                 merger_penalty: float = 0.,
                 merger_per_subject: bool = False,
                 dropout: float = 0.,
                 dropout_rescale: bool = True,
                 initial_linear: int = 0,
                 initial_depth: int = 1,
                 initial_nonlin: bool = False,
                 subsample_meg_channels: int = 0,
                 dataset_cfg=None,
                 ):
        super().__init__()
        in_channels = dict(in_channels)
        hidden = dict(hidden)

        if set(in_channels.keys()) != set(hidden.keys()):
            raise ValueError("Channels and hidden keys must match "
                             f"({set(in_channels.keys())} and {set(hidden.keys())})")
        self._concatenate = concatenate
        self.out_channels = out_channels

        if gelu:
            activation = nn.GELU
        elif relu_leakiness:
            activation = partial(nn.LeakyReLU, relu_leakiness)
        else:
            activation = nn.ReLU

        assert kernel_size % 2 == 1, "For padding to work, this must be verified"

        self.merger = None
        self.dropout = None
        self.subsampled_meg_channels: tp.Optional[list] = None
        if subsample_meg_channels:
            assert 'meg' in in_channels
            indexes = list(range(in_channels['meg']))
            rng = random.Random(1234)
            rng.shuffle(indexes)
            self.subsampled_meg_channels = indexes[:subsample_meg_channels]

        self.initial_linear = None
        if dropout > 0.:
            self.dropout = ChannelDropout(dropout, dropout_rescale, dataset_cfg=dataset_cfg)
        if merger:
            self.merger = ChannelMerger(
                merger_channels, pos_dim=merger_pos_dim, dropout=merger_dropout,
                usage_penalty=merger_penalty, n_subjects=n_subjects, per_subject=merger_per_subject,
                dataset_cfg=dataset_cfg)
            in_channels["meg"] = merger_channels

        if initial_linear:
            init = [nn.Conv1d(in_channels["meg"], initial_linear, 1)]
            for _ in range(initial_depth - 1):
                init += [activation(), nn.Conv1d(initial_linear, initial_linear, 1)]
            if initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)
            in_channels["meg"] = initial_linear

        self.subject_layers = None
        if subject_layers:
            assert "meg" in in_channels
            meg_dim = in_channels["meg"]
            dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
            self.subject_layers = SubjectLayers(meg_dim, dim, n_subjects, subject_layers_id)
            in_channels["meg"] = dim

        self.subject_embedding = None
        if subject_dim:
            self.subject_embedding = ScaledEmbedding(n_subjects, subject_dim, embedding_scale)
            in_channels["meg"] += subject_dim

        # concatenate inputs if need be
        if concatenate:
            in_channels = {"concat": sum(in_channels.values())}
            hidden = {"concat": sum(hidden.values())}

        # compute the sequences of channel sizes
        sizes = {}
        for name in in_channels:
            sizes[name] = [in_channels[name]]
            sizes[name] += [int(round(hidden[name] * growth ** k)) for k in range(depth)]

        params: tp.Dict[str, tp.Any]
        params = dict(kernel=kernel_size, stride=1,
                      leakiness=relu_leakiness, dropout=conv_dropout, dropout_input=dropout_input,
                      batch_norm=batch_norm, dilation_growth=dilation_growth, groups=groups,
                      dilation_period=dilation_period, skip=skip, post_skip=post_skip, scale=scale,
                      rewrite=rewrite, glu=glu, glu_context=glu_context, glu_glu=glu_glu,
                      activation=activation)

        final_channels = sum([x[-1] for x in sizes.values()])
        self.final = None
        pad = 0
        kernel = 1
        stride = 1

        if linear_out:
            assert not complex_out
            self.final = nn.ConvTranspose1d(final_channels, out_channels, kernel, stride, pad)
        elif complex_out and meg_encoder == 'cnn':
            self.final = nn.Sequential(
                nn.Conv1d(final_channels, 2 * final_channels, 1),
                activation(),
                nn.ConvTranspose1d(2 * final_channels, out_channels, kernel, stride, pad))
        elif complex_out and meg_encoder == 'conformer':
            conformer_dim = tf_hidden
            self.final = nn.Sequential(
                nn.Conv1d(conformer_dim, 2 * conformer_dim, 1),
                activation(),
                nn.ConvTranspose1d(2 * conformer_dim, out_channels, kernel, stride, pad))
        else:
            assert len(sizes) == 1, "if no linear_out, there must be a single branch."
            params['activation_on_last'] = False
            list(sizes.values())[0][-1] = out_channels

        if meg_encoder == 'cnn':
            self.encoders = nn.ModuleDict({name: ConvSequence(channels, **params)
                                           for name, channels in sizes.items()})
        elif meg_encoder == 'conformer':
            self.encoders = nn.ModuleDict({
                name: ConformerEncoder(
                    input_dim=channels[0],    # Input dimension (# of channels - 270)
                    num_layers=num_layers,    # Number of transformer layers
                    d_model=tf_hidden,        # Embedding dimension (internal channel dim)
                    n_heads=nheads,           # Number of attention heads
                    d_ff=tf_hidden * 4,       # Feed-forward dimension
                    conv_kernel=31,           # Convolutional kernel size
                    dropout=tf_dropout,       # Dropout rate
                )
                for name, channels in sizes.items()
            })
        else:
            raise ValueError(f"Unknown MEG encoder: {meg_encoder}.")

    def forward(self, inputs, subjects, batch):
        length = next(iter(inputs.values())).shape[-1]  # length of any of the inputs

        if self.subsampled_meg_channels is not None:
            mask = torch.zeros_like(inputs["meg"][:1, :, :1])
            mask[:, self.subsampled_meg_channels] = 1.
            inputs["meg"] = inputs["meg"] * mask

        if self.dropout is not None:
            inputs["meg"] = self.dropout(inputs["meg"], batch)

        if self.merger is not None:
            inputs["meg"] = self.merger(inputs["meg"], subjects, batch)

        if self.initial_linear is not None:
            inputs["meg"] = self.initial_linear(inputs["meg"])

        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)

        if self.subject_embedding is not None:
            emb = self.subject_embedding(subjects)[:, :, None]
            inputs["meg"] = torch.cat([inputs["meg"], emb.expand(-1, -1, length)], dim=1)

        if self._concatenate:
            input_list = [x[1] for x in sorted(inputs.items())]
            inputs = {"concat": torch.cat(input_list, dim=1)}

        encoded = {}
        for name, x in inputs.items():
            encoded[name] = self.encoders[name](x)
        inputs = [x[1] for x in sorted(encoded.items())]
        x = torch.cat(inputs, dim=1)
        if self.final is not None:
            x = self.final(x)
        assert x.shape[-1] >= length
        return x[:, :, :length]
