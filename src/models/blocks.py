import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.models.gated_deltaproduct import GatedDeltaProductConfig
from fla.models.gated_deltaproduct.modeling_gated_deltaproduct import (
    GatedDeltaProductBlock,
)


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


class GatedDeltaNetEncoder(nn.Module):
    """
    GatedDeltaNet encoder using GatedDeltaProductBlock for sequence modeling.
    """

    def __init__(self, layer_idx, token_embed_dim, num_heads=4, **kwargs):
        super().__init__()
        config = GatedDeltaProductConfig(
            attn_mode="chunk",
            hidden_size=token_embed_dim,
            expand_v=1.0,
            use_gate=False,
            use_short_conv=True,
            conv_size=4,
            head_dim=token_embed_dim // num_heads,
            num_heads=num_heads,
            allow_neg_eigval=True,
            use_forget_gate=True,
        )
        self.encoder_layer = GatedDeltaProductBlock(layer_idx=layer_idx, config=config)

    def forward(self, x, initial_state=None):
        """
        Forward pass through the GatedDeltaProductBlock.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor of same shape as input
        """
        x, last_hidden_state, _ = self.encoder_layer(
            x, output_attentions=True, initial_state=initial_state
        )
        return x, last_hidden_state
