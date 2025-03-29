import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers.delta_net import DeltaNet
from fla.layers.gated_deltanet import GatedDeltaNet

from src.utils.utils import device


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
        super(SinPositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(sin_pos_const) / d_model)
        )
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0).to(device)

    def forward(self, x):
        return self.encoding[:, : x.size(1)].detach()


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
                in_channels=out_channels,  # Total number of channels from all convolutions
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, x):
        # x is expected to be of shape (batch_size, in_channels, sequence_length)
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


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        norm=True,
        norm_type="layernorm",
        residual=False,
        name="EncoderBlock",
        enc_type="GatedDeltaNet",
        enc_conv=False,
        enc_conv_kernel=5,
        enc_conv_dilation=0,
        head_dim=256,
        num_heads=4,
        block_expansion=2.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.name = name
        self.norm = norm

        if enc_type == "GatedDeltaNet":
            self.encoder_layer = GatedDeltaNet(
                mode="chunk",
                hidden_size=embed_dim,
                expand_v=block_expansion,
                head_dim=head_dim,
                num_heads=num_heads,
                use_gate=True,
                use_short_conv=True,
                conv_size=4,
                norm_first=norm,
                norm_eps=1e-6,
            )

        elif enc_type == "DeltaNet":
            self.encoder_layer = DeltaNet(
                mode="chunk",
                hidden_size=embed_dim,
                expand_k=1.0,
                expand_v=block_expansion,
                head_dim=head_dim,
                num_heads=num_heads,
                use_gate=True,
                use_short_conv=True,
                allow_neg_eigval=True,
                conv_size=4,
            )

        if enc_conv:
            self.stage_2_layer = DilatedConv1dBlock(
                embed_dim,
                embed_dim,
                enc_conv_kernel,
                enc_conv_dilation,
                single_conv=False,
            )
            self.stage_2_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU()
            )

        if self.norm:
            if norm_type == "layernorm":
                self.norm_layer_1 = nn.LayerNorm(embed_dim)
                self.norm_layer_2 = nn.LayerNorm(embed_dim)
            elif norm_type == "rmsnorm":
                self.norm_layer_1 = SimpleRMSNorm(embed_dim)
                self.norm_layer_2 = SimpleRMSNorm(embed_dim)

        self.residual = residual

    def forward(self, x):
        if self.norm:
            x_enc, _, _ = self.encoder_layer(self.norm_layer_1(x))
        else:
            x_enc, _, _ = self.encoder_layer(x)

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


class ConcatLayer(nn.Module):
    def __init__(self, dim=1, name=None):
        super().__init__()
        self.dim = dim
        self.name = name

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)
