import torch
import torchaudio
from pathlib import Path
from audio_tokenizer.modeling_audio_vae import AudioVAE


class MingAudioProcessor:
    def __init__(self, model_name='inclusionAI/MingTok-Audio', device='cuda'):
        """
        Initialize the MingAudioProcessor

        Args:
            model_name: HuggingFace model name or local path
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        """Load the AudioVAE model"""
        self.model = AudioVAE.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, input_audio_path, output_audio_path):
        """
        Generate audio reconstruction

        Args:
            input_audio_path: Path to input audio file
            output_audio_path: Path to save output audio file

        Returns:
            output_waveform: Generated audio tensor
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        input_path = Path(input_audio_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")

        waveform, sr = torchaudio.load(str(input_path), backend='soundfile')

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        sample = {
            'waveform': waveform.to(self.device),
            'waveform_length': torch.tensor([waveform.size(-1)]).to(self.device)
        }

        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                latent, frame_num = self.model.encode_latent(**sample)
                output_waveform = self.model.decode(latent)

        output_path = Path(output_audio_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), output_waveform[0].cpu()[0], sample_rate=16000)

        return output_waveform
