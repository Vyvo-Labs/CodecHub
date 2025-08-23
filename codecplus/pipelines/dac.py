import torchaudio

from codecplus.codecs.dac.model import DAC


class DACPipeline:
    def __init__(
        self,
        model_path=None,
        config_path=None,
        device="cpu",
        sample_rate=44100,
        n_mels=80,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        self.dac_model = None
        if self.dac_model is None:
            self.load_model()

    def load_model(self):
        dac_model = DAC(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels
        )
        dac_model = dac_model.to(self.device)

        self.dac_model = dac_model
        return dac_model

    def encode(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)

        wav = wav.to(self.device)
        latents = self.dac_model.encode(wav)

        self.latents = latents
        return latents

    def decode(self, audio_outpath, latents=None):
        if latents is None:
            latents = self.latents

        audio_out = self.dac_model.decode(latents)

        torchaudio.save(
            audio_outpath,
            audio_out.cpu(),
            sample_rate=self.sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )

        return audio_out
