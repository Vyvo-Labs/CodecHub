import os
import math
from typing import Optional

import torch
import torch.nn as nn
import librosa
import safetensors
import accelerate
from huggingface_hub import snapshot_download

from transformers import AutoTokenizer

from codecplus.codecs.tadicodec.models.tts.tadicodec.infer_utils import build_vocoder_model, build_mel_model
from codecplus.codecs.tadicodec.models.tts.tadicodec.modeling_tadicodec import TaDiCodec


class TaDiCodecPipline(nn.Module):
    def __init__(
        self,
        cfg,
        model_path: str,
        device: torch.device,
        tokenizer_path: Optional[str] = None,
        vocoder_ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.model_path = model_path
        self.device = device

        # tokenizer
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_path)
            if tokenizer_path is not None
            else os.path.join(model_path, "text_tokenizer")
        )

        # mel feature extractor
        self.mel_model = build_mel_model(cfg, device)

        # main model
        tadiconfig = cfg.model.tadicodec

        self.tadicodec = TaDiCodec(cfg=tadiconfig)
        safetensors.torch.load_model(self.tadicodec, model_path, strict=False)
        self.tadicodec.to(torch.float32)
        self.tadicodec.to(device)
        self.tadicodec.eval()

        # vocoder
        self.vocoder_model = build_vocoder_model(cfg, device)
        v_path = (
            vocoder_ckpt_path
            if vocoder_ckpt_path
            else os.path.join(model_path, "vocoder")
        )
        accelerate.load_checkpoint_and_dispatch(self.vocoder_model, v_path)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str = "./ckpt/TaDiCodec",
        device: Optional[torch.device] = None,
        auto_download: bool = True,
    ):
        """Create a pipeline from a checkpoint directory or Hugging Face model ID.

        Expected structure under `ckpt_dir`:
          - config.json                # model and preprocess config
          - model.safetensors          # TaDiCodec weights
          - vocoder/                   # directory containing vocoder weights
              model.safetensors or other *.safetensors
          - text_tokenizer/               # directory containing text tokenizer
              tokenizer.json
              tokenizer_config.json

        Args:
            ckpt_dir: Directory containing `config.json`, `model.safetensors`, and `vocoder/`, or Hugging Face model ID.
            device: Device to place models on. Defaults to CUDA if available else CPU.
            auto_download: Whether to automatically download models from Hugging Face if not found locally

        Returns:
            TaDiCodecPipline
        """
        import os
        import glob
        from codecplus.codecs.tadicodec.utils.util import load_config

        resolved_device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Resolve checkpoint directory
        resolved_ckpt_dir = cls._resolve_model_path(
            ckpt_dir, auto_download=auto_download, model_type="tadicodec"
        )

        # Load config
        config_path = os.path.join(resolved_ckpt_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        cfg = load_config(config_path, lowercase=False)

        # Resolve main model weights
        model_weights_path = os.path.join(resolved_ckpt_dir, "model.safetensors")

        # Resolve vocoder weights
        vocoder_ckpt_path = os.path.join(resolved_ckpt_dir, "vocoder")

        text_tokenizer_dir = os.path.join(resolved_ckpt_dir, "text_tokenizer")

        return cls(
            cfg=cfg,
            model_path=model_weights_path,
            device=resolved_device,
            vocoder_ckpt_path=vocoder_ckpt_path,
            tokenizer_path=text_tokenizer_dir,
        )

    @staticmethod
    def _resolve_model_path(
        model_path: str, auto_download: bool = True, model_type: str = "tadicodec"
    ) -> str:
        """
        Resolve model path, downloading from Hugging Face if necessary

        Args:
            model_path: Local path or Hugging Face model ID
            auto_download: Whether to auto-download from HF
            model_type: Type of model ("tadicodec")

        Returns:
            Resolved local path
        """
        # If it's already a local path and exists, return as is
        if os.path.exists(model_path):
            return model_path

        # If it looks like a Hugging Face model ID (contains '/')
        if "/" in model_path and auto_download:
            print(f"Downloading {model_type} model from Hugging Face: {model_path}")
            try:
                # Download to cache directory
                cache_dir = os.path.join(
                    os.path.expanduser("~"), ".cache", "huggingface", "hub"
                )
                downloaded_path = snapshot_download(
                    repo_id=model_path,
                    cache_dir=cache_dir,
                    local_dir_use_symlinks=False,
                )
                print(
                    f"Successfully downloaded {model_type} model to: {downloaded_path}"
                )
                return downloaded_path
            except Exception as e:
                print(f"Failed to download {model_type} model from Hugging Face: {e}")
                raise ValueError(
                    f"Could not download {model_type} model from {model_path}"
                )

        # If it's a local path that doesn't exist
        if not os.path.exists(model_path):
            if auto_download:
                raise ValueError(
                    f"Model path does not exist: {model_path}. Set auto_download=True to download from Hugging Face."
                )
            else:
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

        return model_path

    @torch.no_grad()
    def __call__(
        self,
        text: Optional[str] = None,
        speech_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_speech_path: Optional[str] = None,
        n_timesteps: int = 32,
        return_code: bool = False,
        cfg_scale: float = 2.0,
    ):
        # tokenize text
        text_input_ids = self.tokenize_text(text, prompt_text)

        # extract mel features
        target_mel = self.extract_mel_feature(speech_path)
        prompt_mel = (
            self.extract_mel_feature(prompt_speech_path) if prompt_speech_path else None
        )

        # encode to codes from mel
        if prompt_mel is not None:
            vq_emb, indices = self.encode(torch.cat([prompt_mel, target_mel], dim=1))
        else:
            vq_emb, indices = self.encode(target_mel)

        if return_code:
            return indices

        # decode mel from codes + optional text/prompt
        rec_mel = self.decode(
            vq_emb=vq_emb,
            text_token_ids=text_input_ids,
            prompt_mel=(
                prompt_mel if prompt_mel is not None else target_mel[:, : 50 * 3]
            ),
            n_timesteps=n_timesteps,
            cfg=cfg_scale,
            rescale_cfg=0.75,
        )

        # vocoder
        rec_audio = (
            self.vocoder_model(rec_mel.transpose(1, 2)).detach().cpu().numpy()[0][0]
        )
        return rec_audio

    def tokenize_text(
        self, text: Optional[str] = None, prompt_text: Optional[str] = None
    ):
        if self.tokenizer is None or text is None:
            return None
        if prompt_text is not None:
            text_token_ids = self.tokenizer(
                prompt_text + text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
        else:
            text_token_ids = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
        return text_token_ids

    @torch.no_grad()
    def extract_mel_feature(self, speech_path: Optional[str]):
        assert speech_path is not None and os.path.exists(
            speech_path
        ), f"Invalid speech_path: {speech_path}"
        speech = librosa.load(speech_path, sr=24000)[0]
        speech = torch.tensor(speech).to(self.device).unsqueeze(0)
        mel_feature = self.mel_model(speech)  # (B, n_mels, T)
        mel_feature = mel_feature.transpose(1, 2)  # (B, T, n_mels)
        mel_feature = (mel_feature - self.cfg.preprocess.mel_mean) / math.sqrt(
            self.cfg.preprocess.mel_var
        )
        return mel_feature

    @torch.no_grad()
    def encode(self, mel_feat: torch.Tensor):
        vq_emb, indices = self.tadicodec.encode(
            mel_feat, torch.ones(mel_feat.shape[0], mel_feat.shape[1]).to(self.device)
        )
        return vq_emb, indices

    @torch.no_grad()
    def decode(
        self,
        vq_emb: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        text_token_ids: Optional[torch.Tensor] = None,
        prompt_mel: Optional[torch.Tensor] = None,
        n_timesteps: int = 32,
        cfg: float = 1.0,
        rescale_cfg: float = 0.75,
    ):
        rec_mel = self.tadicodec.reverse_diffusion(
            vq_emb=vq_emb,
            indices=indices,
            text_ids=text_token_ids,
            prompt_mel=prompt_mel,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )
        return rec_mel
