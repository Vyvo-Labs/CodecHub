import math

import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

EPS = torch.finfo(torch.float32).eps


def generate_padding_mask(inputs, lengths=None):
    max_length = inputs.size(0)
    device = inputs.device
    return generate_padding_mask_by_size(max_length, lengths, device)


def generate_padding_mask_by_size(max_length, lengths, device="cpu"):
    if lengths is None:
        return None

    lengths = torch.as_tensor(lengths)
    mask = lengths[:, None] <= torch.arange(max_length, device=lengths.device)[None, :]
    mask = mask.to(device)
    return mask


def calculate_mean_var_postprocess(cmvn_mean, cmvn_var):
    cmvn_mean = torch.from_numpy(cmvn_mean).float()
    cmvn_var = torch.from_numpy(cmvn_var).float()
    eps = torch.tensor(torch.finfo(torch.float32).eps)
    cmvn_scale = 1.0 / torch.max(eps, torch.sqrt(cmvn_var))
    return cmvn_mean, cmvn_scale


def calculate_mean_var(cmvn_stats_0, cmvn_stats_1, cmvn_count):
    cmvn_mean = cmvn_stats_0 / cmvn_count
    cmvn_var = cmvn_stats_1 / cmvn_count - cmvn_mean * cmvn_mean
    return cmvn_mean, cmvn_var


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2**(x - 1).bit_length()


def inverse_mel_scale_scalar(mel_freq: float) -> float:
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


def inverse_mel_scale(mel_freq: Tensor) -> Tensor:
    return 700.0 * ((mel_freq / 1127.0).exp() - 1.0)


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


class Fbank(nn.Module):
    """
    class for GPU batch fbank computation
    modified from https://github.com/pytorch/audio/blob/main/torchaudio/compliance/kaldi.py#L514
    main modifications:
        1. support batch computation
        2. pre-compute filter banks
        3. simplify the process and remove process we do not use
    refer to kaldi for the meaning of the config parameters
    """

    def __init__(self, conf):
        super(Fbank, self).__init__()
        self.dither = conf["dither"]
        self.frame_length = frame_length = conf["frame_length"]
        self.frame_shift = frame_shift = conf["frame_shift"]
        self.preemphasis = preemphasis = conf["preemphasis"]
        self.freq = freq = conf["freq"]
        high_freq = conf["high_freq"]
        low_freq = conf["low_freq"]
        self.num_mel_bins = num_mel_bins = conf["num_mel_bins"]

        assert self.freq in [8000, 16000], "freq should be 8000 or 16000"

        self.window_shift = window_shift = int(self.freq * self.frame_shift / 1000)
        self.window_size = window_size = int(self.freq * self.frame_length / 1000)
        self.padded_window_size = padded_window_size = next_power_of_2(self.window_size)

        window = torch.hann_window(self.window_size, periodic=False, dtype=torch.float32).pow(0.85)
        self.register_buffer("window", window)

        # Get mel filter banks
        num_fft_bins = padded_window_size // 2
        nyquist = freq / 2
        self.high_freq = high_freq = high_freq + nyquist if high_freq < 0 else high_freq
        fft_bin_width = freq / padded_window_size

        mel_low_freq = mel_scale_scalar(low_freq)
        mel_high_freq = mel_scale_scalar(high_freq)
        mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_mel_bins + 1)
        bins = torch.arange(num_mel_bins).unsqueeze(1)
        left_mel = mel_low_freq + bins * mel_freq_delta
        center_mel = mel_low_freq + (bins + 1.0) * mel_freq_delta
        right_mel = mel_low_freq + (bins + 2.0) * mel_freq_delta

        mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)

        # size (num_mel_bins, num_fft_bins)
        up_slope = (mel - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel) / (right_mel - center_mel)
        mel_banks = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
        mel_banks = F.pad(mel_banks, (0, 1), mode="constant", value=0)
        mel_banks = mel_banks.t()
        self.register_buffer("mel_banks", mel_banks)

        eps = torch.tensor(torch.finfo(torch.float32).eps)
        self.register_buffer("eps", eps)

        # need padding for lower feat
        self.pad_size = 0
        if "padded_num_mel_bins" in conf:
            self.pad_size = conf["padded_num_mel_bins"] - self.num_mel_bins


    def forward(self, batch_wav, input_lens):
        """
        Args:
            batch_wav: batched wav, shape (batch, wav_len)
            input_lens: shape (batch,)
        Returns:
            batch_fbank: shape (batch, frame_len, num_mel_bins)
            output_lens: shape (batch, )
        """
        with torch.no_grad():
            b, wav_len = batch_wav.size()

            if wav_len < self.window_size:
                feats = torch.empty((b, 0, self.num_mel_bins), dtype=batch_wav.dtype, device=batch_wav.device)
                input_lens = torch.zeros(b, dtype=input_lens.dtype, device=input_lens.device)
                return feats, input_lens

            frame_len = 1 + (wav_len - self.window_size) // self.window_shift
            # batch_wav should be contiguous before using as_strided
            batch_wav = batch_wav.contiguous()
            # (n, frame_len, window_size)
            frames = batch_wav.as_strided((b, frame_len, self.window_size), (wav_len, self.window_shift, 1))
            if self.dither != 0.0:
                rand_gauss = torch.randn(frames.shape, device=frames.device, dtype=frames.dtype)
                frames = frames + rand_gauss * self.dither

            # (n, frame_len, window_size + 1)
            padded = F.pad(frames, (1, 0), mode="replicate")
            frames = frames - self.preemphasis * padded[:, :, :-1]
            frames = frames * self.window
            if self.padded_window_size != self.window_size:
                padding_right = self.padded_window_size - self.window_size
                frames = F.pad(frames, (0, padding_right), mode="constant", value=0)
            # spectrum
            spec = torch.fft.rfft(frames).abs()
            # power, (n, frame_len, num_fft_bins)
            spec = spec.pow(2)

            mel_energy = torch.matmul(spec, self.mel_banks)
            mel_energy = torch.max(mel_energy, self.eps).log()
            # input_lens = (input_lens - self.window_size) // self.window_shift + 1
            input_lens = torch.div(input_lens - self.window_size, self.window_shift, rounding_mode='trunc') + 1

            # padding feature when mixband is 8k
            if self.pad_size > 0:
                mel_energy = F.pad(mel_energy, (0, self.pad_size))

            return mel_energy, input_lens


class Cmvn(nn.Module):
    """
    class for GPU cmvn computation
    """

    def __init__(self, cmvn_file):
        super(Cmvn, self).__init__()

        cmvn_stats = np.load(cmvn_file)
        cmvn_count = cmvn_stats[0][-1]

        cmvn_mean, cmvn_var = calculate_mean_var(cmvn_stats[0][:-1], cmvn_stats[1][:-1], cmvn_count)
        cmvn_mean, cmvn_scale = calculate_mean_var_postprocess(cmvn_mean, cmvn_var)

        self.register_buffer("cmvn_mean", cmvn_mean)
        self.register_buffer("cmvn_scale", cmvn_scale)

    @torch.no_grad()
    def forward(self, feats, input_lens):
        B, T, D = feats.size()
        if T == 0:
            return feats, input_lens

        feats = (feats - self.cmvn_mean) * self.cmvn_scale

        return feats, input_lens


class FeatureExtractor(nn.Module):
    """
    class for feature computation which include fbank and cmvn
    """
    def __init__(self, conf):
        super(FeatureExtractor, self).__init__()

        self.fbank = Fbank(conf["fbank"])
        self.cmvn = Cmvn(conf["cmvn_file"])

    @torch.no_grad()
    def forward(self, feats, input_lens):
        feats, input_lens = self.fbank(feats, input_lens)
        feats, input_lens = self.cmvn(feats, input_lens)

        return feats, input_lens