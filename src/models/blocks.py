import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers.delta_net import DeltaNet
from fla.layers.gated_deltanet import GatedDeltaNet


class PositionExpansion(nn.Module):
    """
    Creates positional embeddings for time features.
    """

    def __init__(self, periods: int, freqs: int):
        super().__init__()
        # Channels could be ceiling(log_2(periods))
        self.periods = periods
        self.channels = freqs * 2

        # Create position encoding
        pos_encoding = np.hstack(
            [
                np.fromfunction(
                    lambda i, j: np.sin(np.pi / periods * (2**j) * (i - 1)),
                    (periods + 1, freqs),
                ),
                np.fromfunction(
                    lambda i, j: np.cos(np.pi / periods * (2**j) * (i - 1)),
                    (periods + 1, freqs),
                ),
            ]
        )
        self.embedding = torch.tensor(
            pos_encoding, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def forward(self, tc: torch.Tensor):
        flat = tc.view(-1)  # Flatten completely
        embedded = self.embedding.index_select(0, flat.to(torch.long))
        out_shape = tc.shape + (self.channels,)  # Add channel dimension
        return embedded.view(*out_shape)


class SimpleRMSNorm(nn.Module):
    """
    SimpleRMSNorm

    Args:
        dim (int): dimension of the embedding

    Usage:
    We can use SimpleRMSNorm as a layer in a neural network as follows:
        >>> x = torch.randn(1, 10, 512)
        >>> simple_rms_norm = SimpleRMSNorm(dim=512)
        >>> simple_rms_norm(x).shape
        torch.Size([1, 10, 512])
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, x):
        """Forward method of SimpleRMSNorm"""
        return F.normalize(x, dim=-1) * self.scale


class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model=36, max_len=5000, sin_pos_const=10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.sin_pos_const = sin_pos_const

    def forward(self, x, num_channels):
        batch_size, seq_len, _ = x.shape
        encoding = torch.zeros(
            batch_size, seq_len, num_channels, self.d_model, device=x.device
        )
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device).float()
            * -(math.log(self.sin_pos_const) / self.d_model)
        )
        for c in range(num_channels):
            # Modulate frequencies by channel index
            channel_offset = (c + 1) * 0.1
            encoding[:, :, c, 0::2] = torch.sin(positions * div_term * channel_offset)
            encoding[:, :, c, 1::2] = torch.cos(positions * div_term * channel_offset)
        return encoding.detach()


class DilatedConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        max_dilation=3,
        single_conv=False,
        conv_gelu=True,
    ):
        super(DilatedConv1dBlock, self).__init__()
        self.conv_gelu = conv_gelu
        self.single_conv = single_conv
        if self.single_conv:
            padding = (kernel_size - 1) * 2**max_dilation
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=2**max_dilation,
                bias=True,
            )
        else:
            self.conv = nn.ModuleList()
            conv_out_channels = out_channels // (max_dilation + 1)
            for dilation in range(max_dilation + 1):
                padding = (kernel_size - 1) * 2**dilation
                self.conv.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=conv_out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=2**dilation,
                        bias=True,
                    )
                )
            self.inception_conv = nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, x):
        x = x.transpose(1, 2)
        seq_len = x.shape[-1]
        if self.single_conv:
            x = self.conv(x)
            if self.conv_gelu:
                x = F.gelu(x)
        else:
            if self.conv_gelu:
                x_list = [
                    F.gelu(conv_layer(x))[:, :, :seq_len] for conv_layer in self.conv
                ]
            else:
                x_list = [conv_layer(x)[:, :, :seq_len] for conv_layer in self.conv]
            x = torch.cat(x_list, dim=1)
            x = self.inception_conv(x)
        return x.transpose(1, 2)


class ConcatLayer(nn.Module):
    def __init__(self, dim=1, name=None):
        super().__init__()
        self.dim = dim
        self.name = name

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class BaseEncoder(nn.Module):
    """Base class for encoders with common functionality"""

    def __init__(
        self,
        token_embed_dim=1024,
        norm=True,
        norm_type="layernorm",
        residual=False,
        enc_conv=False,
        dilated_conv_kernel_size=5,
        dilated_conv_max_dilation=0,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm
        self.residual = residual

        if self.norm:
            if norm_type == "layernorm":
                self.norm_layer_1 = nn.LayerNorm(token_embed_dim)
                self.norm_layer_2 = nn.LayerNorm(token_embed_dim)
            elif norm_type == "rmsnorm":
                self.norm_layer_1 = SimpleRMSNorm(token_embed_dim)
                self.norm_layer_2 = SimpleRMSNorm(token_embed_dim)

        if enc_conv:
            self.stage_2_layer = DilatedConv1dBlock(
                token_embed_dim,
                token_embed_dim,
                dilated_conv_kernel_size,
                dilated_conv_max_dilation,
                single_conv=False,
            )
        else:
            self.stage_2_layer = nn.Sequential(
                nn.Linear(token_embed_dim, token_embed_dim), nn.GELU()
            )

    def setup_encoder_layer(self, **kwargs):
        raise NotImplementedError("Subclasses must implement setup_encoder_layer")

    def forward(self, x):
        # Apply normalization before encoder if enabled
        if self.norm:
            x_enc, _, _ = self.encoder_layer(self.norm_layer_1(x))
        else:
            x_enc, _, _ = self.encoder_layer(x)

        # Apply residual connection if enabled
        if self.residual:
            x = x + x_enc
        else:
            x = x_enc

        if self.norm:
            x_out = self.stage_2_layer(self.norm_layer_2(x))
        else:
            x_out = self.stage_2_layer(x)

        if self.residual:
            x_out = x_out + x

        return x_out


class GatedDeltaNetEncoder(BaseEncoder):
    def __init__(
        self, token_embed_dim, head_dim=256, num_heads=4, block_expansion=2.0, **kwargs
    ):
        super().__init__(token_embed_dim=token_embed_dim, **kwargs)
        self.encoder_layer = GatedDeltaNet(
            mode="chunk",
            hidden_size=token_embed_dim,
            expand_v=block_expansion,
            head_dim=head_dim,
            num_heads=num_heads,
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
            norm_first=self.norm,
            norm_eps=1e-6,
            allow_neg_eigval=False,
        )


class DeltaNetEncoder(BaseEncoder):
    def __init__(
        self, token_embed_dim, head_dim=256, num_heads=4, block_expansion=2.0, **kwargs
    ):
        super().__init__(token_embed_dim=token_embed_dim, **kwargs)
        self.encoder_layer = DeltaNet(
            mode="chunk",
            hidden_size=token_embed_dim,
            expand_k=1.0,
            expand_v=block_expansion,
            head_dim=head_dim,
            num_heads=num_heads,
            use_gate=True,
            use_short_conv=True,
            allow_neg_eigval=True,
            conv_size=4,
        )


class EncoderFactory:
    @staticmethod
    def create_encoder(encoder_type, **encoder_config):
        if encoder_type == "GatedDeltaNet":
            return GatedDeltaNetEncoder(**encoder_config)
        elif encoder_type == "DeltaNet":
            return DeltaNetEncoder(**encoder_config)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
