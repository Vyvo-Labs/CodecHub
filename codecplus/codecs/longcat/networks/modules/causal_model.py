import math
from torch import nn

from codecplus.codecs.longcat.networks.modules.layers import Snake1d, WNConv1d, WNConvTranspose1d


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        # nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.orthogonal_(m.weight)  # use this to avoid gradient boom.
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2 * 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.ZeroPad1d((pad, 0)),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=0),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1])
        # 按照设计，pad应该一直等于0
        if pad > 0:
            x = x[..., pad:]
        return x + y


class EncoderBlock(nn.Module):

    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.ZeroPad1d((math.ceil(stride / 2) * 2, 0)),
            WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=0)
        )

    def forward(self, x):
        return self.block(x)


class CausalEncoder(nn.Module):

    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()

        # Create first convolution
        self.block = [nn.ZeroPad1d((6, 0)), WNConv1d(1, d_model, kernel_size=7, padding=0)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            nn.ZeroPad1d((2, 0)),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=0),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

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
                kernel_size=stride,
                stride=stride,
                padding=0,
                output_padding=0,  #stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class CausalDecoder(nn.Module):

    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [nn.ZeroPad1d((6, 0)), WNConv1d(input_channel, channels, kernel_size=7, padding=0)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2**(i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            nn.ZeroPad1d((6, 0)),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CausalDecoderBlock(nn.Module):

    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=stride,
                stride=stride,
                padding=0,
                output_padding=0,  #stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9), 
        )

    def forward(self, x):
        return self.block(x)