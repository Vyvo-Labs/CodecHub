from codecplus.codecs.tadicodec.models.codec.melvqgan.melspec import MelSpectrogram
from codecplus.codecs.tadicodec.models.codec.amphion_codec.vocos import Vocos


def build_vocoder_model(cfg, device):
    vocoder_model = Vocos(cfg=cfg.model.vocos)
    vocoder_model.eval()
    vocoder_model.to(device)
    return vocoder_model


def build_mel_model(cfg, device):
    mel_model = MelSpectrogram(
        sampling_rate=cfg.preprocess.sample_rate,
        n_fft=cfg.preprocess.n_fft,
        num_mels=cfg.preprocess.num_mels,
        hop_size=cfg.preprocess.hop_size,
        win_size=cfg.preprocess.win_size,
        fmin=cfg.preprocess.fmin,
        fmax=cfg.preprocess.fmax,
    )
    mel_model.eval()
    mel_model.to(device)
    return mel_model
