import os
from typing import Any, Dict, Tuple, Union

import torch
import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from torch import nn

from codecplus.codecs.wav_tokenizer.decoder.feature_extractors import (
    EncodecFeatures,
    FeatureExtractor,
)
from codecplus.codecs.wav_tokenizer.decoder.heads import FourierHead
from codecplus.codecs.wav_tokenizer.decoder.models import Backbone


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """
    Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class WavTokenizer(nn.Module):
    """
    The Vocos class represents a Fourier-based neural vocoder for audio synthesis.

    This class is primarily designed for inference, with support for loading from pretrained model
    checkpoints. It consists of three main components: a feature extractor, a backbone, and a head.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config_path: str) -> "WavTokenizer":
        """Class method to create a new Vocos model instance from hyperparameters stored in a yaml
        configuration file.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_class(args=(), init=config["feature_extractor"])
        backbone = instantiate_class(args=(), init=config["backbone"])
        head = instantiate_class(args=(), init=config["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(cls, repo_id: str) -> "WavTokenizer":
        """Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging
        Face model hub.

        Args:
            repo_id: The Hugging Face repository ID

        Returns:
            WavTokenizer: The loaded model instance

        Note:
            This method searches for config files with .yaml or .yml extensions
            and model files with .ckpt or .bin extensions in the repository.
            It supports both old and new config formats automatically.
        """
        # List all files in the repository
        repo_files = list_repo_files(repo_id=repo_id)

        # Find config file (.yaml or .yml)
        config_file = None
        for file in repo_files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                config_file = file
                break

        if config_file is None:
            raise ValueError(f"No config file (.yaml or .yml) found in repository {repo_id}")

        # Find model checkpoint file (.ckpt or .bin)
        model_file = None
        for file in repo_files:
            if file.endswith('.ckpt') or file.endswith('.bin'):
                model_file = file
                break

        if model_file is None:
            raise ValueError(f"No model file (.ckpt or .bin) found in repository {repo_id}")

        # Download files
        config_path = hf_hub_download(repo_id=repo_id, filename=config_file)
        model_path = hf_hub_download(repo_id=repo_id, filename=model_file)

        # Load config to check format
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Determine config format and load model accordingly
        if "model" in config and "init_args" in config["model"]:
            # New format: use from_hparams0802
            model = cls.from_hparams0802(config_path)
        else:
            # Old format: use from_hparams
            model = cls.from_hparams(config_path)

        # Load checkpoint
        state_dict = torch.load(model_path, map_location="cpu")

        # Handle .ckpt format (PyTorch Lightning checkpoint)
        if model_file.endswith('.ckpt'):
            if 'state_dict' in state_dict:
                state_dict_raw = state_dict['state_dict']
                # Filter only relevant keys for the model
                state_dict = {}
                for k, v in state_dict_raw.items():
                    if (
                        k.startswith("backbone.")
                        or k.startswith("head.")
                        or k.startswith("feature_extractor.")
                    ):
                        state_dict[k] = v
            else:
                # If no 'state_dict' key, assume the checkpoint is already in the correct format
                pass

        # For .bin format with old config, add encodec parameters
        if model_file.endswith('.bin') and isinstance(model.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in model.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)

        model.load_state_dict(state_dict)
        model.eval()
        return model

    @classmethod
    def from_hparams0802(cls, config_path: str) -> "WavTokenizer":
        """Class method to create a new Vocos model instance from hyperparameters stored in a yaml
        configuration file.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_class(
            args=(), init=config["model"]["init_args"]["feature_extractor"]
        )
        backbone = instantiate_class(
            args=(), init=config["model"]["init_args"]["backbone"]
        )
        head = instantiate_class(args=(), init=config["model"]["init_args"]["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained0802(self, config_path, model_path):
        """Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging
        Face model hub.
        """
        model = self.from_hparams0802(config_path)
        state_dict_raw = torch.load(model_path, map_location="cpu")["state_dict"]
        state_dict = dict()
        for k, v in state_dict_raw.items():
            if (
                k.startswith("backbone.")
                or k.startswith("head.")
                or k.startswith("feature_extractor.")
            ):
                state_dict[k] = v
        # if isinstance(model.feature_extractor, EncodecFeatures):
        #     encodec_parameters = {
        #         "feature_extractor.encodec." + key: value
        #         for key, value in model.feature_extractor.encodec.state_dict().items()
        #     }
        #     state_dict.update(encodec_parameters)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @classmethod
    def from_pretrained0911(self, config_path, model_folder_path):
        """Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging
        Face model hub.
        """
        model = self.from_hparams0802(config_path)

        models = os.listdir(model_folder_path)
        val_loss = []
        for item in models:
            if not item.startswith("vocos_"):
                continue
            val_loss.append(item[-11:-5])
        val_loss.sort()
        val_loss = val_loss[:3]  # 取前3性能较好的模型平均
        state_dict = dict()
        state_dicts = []
        for item in models:
            if not item.startswith("vocos_"):
                continue
            ll = item[-11:-5]
            if ll not in val_loss:
                continue
            model_path = model_folder_path + "/" + item
            state_dict_raw = torch.load(model_path, map_location="cpu")["state_dict"]
            state_dict_single = dict()
            for k, v in state_dict_raw.items():
                if (
                    k.startswith("backbone.")
                    or k.startswith("head.")
                    or k.startswith("feature_extractor.")
                ):
                    state_dict_single[k] = v
            state_dicts.append(state_dict_single)
        for kk in state_dicts[0].keys():
            vv = state_dicts[0][kk]
            for i in range(1, len(state_dicts)):
                ss = state_dicts[i]
                vv += ss[kk]
            vm = vv / len(state_dicts)
            state_dict[kk] = vm
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, bandwidth_id: torch.Tensor = None, **kwargs: Any) -> torch.Tensor:
        """
        Method to run a copy-synthesis from audio waveform. The feature extractor first processes the audio
        input, which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (T,) or (B, T),
                                        where B is the batch size and T is the waveform length.
            bandwidth_id: Bandwidth ID for encoding. If None, defaults to 0 (first bandwidth)
            **kwargs: Additional arguments

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        # Ensure audio has batch dimension
        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(0)  # (T,) -> (B, T)

        if bandwidth_id is None:
            bandwidth_id = torch.tensor([0], device=audio_input.device)
        features, _, _ = self.feature_extractor(audio_input, bandwidth_id=bandwidth_id, **kwargs)  # 0818
        audio_output = self.decode(features, bandwidth_id=bandwidth_id, **kwargs)
        return audio_output

    # 0818
    @torch.inference_mode()
    def encode(self, audio_input: torch.Tensor, bandwidth_id: torch.Tensor = None, **kwargs: Any) -> torch.Tensor:
        """
        Encode audio to features and discrete codes.

        Args:
            audio_input: Input audio tensor of shape (T,) or (B, T)
            bandwidth_id: Bandwidth ID for encoding. If None, defaults to 0 (first bandwidth)
            **kwargs: Additional arguments

        Returns:
            Tuple of (features, discrete_codes)
        """
        # Ensure audio has batch dimension
        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(0)  # (T,) -> (B, T)

        if bandwidth_id is None:
            bandwidth_id = torch.tensor([0], device=audio_input.device)
        features, discrete_codes, _ = self.feature_extractor(audio_input, bandwidth_id=bandwidth_id, **kwargs)
        return features, discrete_codes

    # 0818
    @torch.inference_mode()
    def encode_infer(self, audio_input: torch.Tensor, bandwidth_id: torch.Tensor = None, **kwargs: Any) -> torch.Tensor:
        """
        Encode audio to features and discrete codes using inference mode.

        Args:
            audio_input: Input audio tensor of shape (T,) or (B, T)
            bandwidth_id: Bandwidth ID for encoding. If None, defaults to 0 (first bandwidth)
            **kwargs: Additional arguments

        Returns:
            Tuple of (features, discrete_codes)
        """
        # Ensure audio has batch dimension
        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(0)  # (T,) -> (B, T)

        if bandwidth_id is None:
            bandwidth_id = torch.tensor([0], device=audio_input.device)
        features, discrete_codes, _ = self.feature_extractor.infer(
            audio_input, bandwidth_id=bandwidth_id, **kwargs
        )
        return features, discrete_codes

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, bandwidth_id: torch.Tensor = None, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.
            bandwidth_id: Bandwidth ID for decoding. If None, defaults to 0 (first bandwidth)
            **kwargs: Additional arguments

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        if bandwidth_id is None:
            bandwidth_id = torch.tensor([0], device=features_input.device)
        x = self.backbone(features_input, bandwidth_id=bandwidth_id, **kwargs)
        audio_output = self.head(x)
        return audio_output

    @torch.inference_mode()
    def codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Transforms an input sequence of discrete tokens (codes) into feature embeddings using the feature
        extractor's codebook weights.

        Args:
            codes (Tensor): The input tensor. Expected shape is (K, L) or (K, B, L),
                            where K is the number of codebooks, B is the batch size and L is the sequence length.

        Returns:
            Tensor: Features of shape (B, C, L), where B is the batch size, C denotes the feature dimension,
                    and L is the sequence length.
        """
        assert isinstance(
            self.feature_extractor, EncodecFeatures
        ), "Feature extractor should be an instance of EncodecFeatures"

        if codes.dim() == 2:
            codes = codes.unsqueeze(1)

        n_bins = self.feature_extractor.encodec.quantizer.bins
        offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
        embeddings_idxs = codes + offsets.view(-1, 1, 1)

        tmp = torch.cat(
            [vq.codebook for vq in self.feature_extractor.encodec.quantizer.vq.layers],
            dim=0,
        )
        # features = torch.nn.functional.embedding(embeddings_idxs, self.feature_extractor.codebook_weights).sum(dim=0)
        features = torch.nn.functional.embedding(embeddings_idxs, tmp).sum(dim=0)
        features = features.transpose(1, 2)

        return features
