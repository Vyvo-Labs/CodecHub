import math

import torch
from torch import nn

from codecplus.codecs.longcat.networks.modules.layers import Snake1d, WNConv1d, WNConvTranspose1d


def get_mask_by_size(audio_input, batch_sizes):
    # audio_input [B, D, T]
    # batch_sizes: [B]
    device = audio_input.device
    B = audio_input.size(0)
    T = audio_input.size(-1)
    D = audio_input.size(1)

    mask = torch.arange(T, device=batch_sizes.device).expand(B, T) < batch_sizes.unsqueeze(1)
    mask = mask.int().unsqueeze(1).to(device)

    return mask


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        # nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.orthogonal_(m.weight)  # use this to avoid gradient boom.
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )
    
    def forward(self, x, batch_sizes=None):
        if batch_sizes is not None:
            mask = get_mask_by_size(x, batch_sizes)

        ori_x = x
        if batch_sizes is not None:
            ori_x = ori_x * mask

        for (i, block_item) in enumerate(self.block):
            if batch_sizes is not None:
                x = x * mask
            x = block_item(x)
        if batch_sizes is not None:
            x = x * mask

        pad = (ori_x.shape[-1] - x.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return ori_x + x


class EncoderBlock(nn.Module):

    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()

        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

        self.stride = stride

    def forward_with_lens(self, x, batch_sizes=None):
        if batch_sizes is not None:
            mask = get_mask_by_size(x, batch_sizes)
        
        for (i, block_item) in enumerate(self.block):
            if batch_sizes is not None:
                x = x * mask
            if isinstance(block_item, ResidualUnit):
                x = block_item(x, batch_sizes)
            else:
                x = block_item(x)
        
        if batch_sizes is not None:
            if self.stride % 2 == 0:
                batch_sizes_new = (batch_sizes // self.stride)
            else:
                batch_sizes_new = ((batch_sizes + 1) // self.stride)
            mask_new = get_mask_by_size(x, batch_sizes_new)
            x = x * mask_new
        
        return x, batch_sizes_new

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):

    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward_with_lens(self, x, batch_sizes=None):
        for (i, block_item) in enumerate(self.block):
            if batch_sizes is not None:
                mask = get_mask_by_size(x, batch_sizes)
                x = x * mask
            if isinstance(block_item, EncoderBlock):
                x, batch_sizes = block_item.forward_with_lens(x, batch_sizes)
            else:
                x = block_item(x)
        
        if batch_sizes is not None:
            mask = get_mask_by_size(x, batch_sizes)
            x = x * mask
        
        return x, batch_sizes

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):

    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()

        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)