import os
import copy
import math
from collections import OrderedDict
from typing import Optional, List, Tuple

import yaml

import torch
from torch import nn, Tensor



from codecplus.codecs.longcat.semantic_tokenizer_general.feature_extractor import FeatureExtractor, generate_padding_mask


EPS = torch.finfo(torch.float32).eps


# For VggTransformer non-streaming structure
class TransformerPreLNEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.0,
                 final_norm=False):
        super(TransformerPreLNEncoderLayer, self).__init__()

        self.final_norm = final_norm
        self.nhead = nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model) if self.final_norm else None

    def forward(self,
                x: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        prev_x = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        x = self.dropout(x)
        x = x + prev_x

        prev_x = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = x + prev_x

        if self.final_norm:
            x = self.norm3(x)

        return x


class Conv2dSubsampling(nn.Module):

    def __init__(self, idim, odim, num_layers=2, stride="2,2"):
        super(Conv2dSubsampling, self).__init__()
        stride = self.stride = list(map(int, stride.split(",")))

        self.num_layers = num_layers
        self.stride = stride

        layers = [("subsampling/pad0", nn.ConstantPad2d((0, 0, 2, 0), 0))]
        layers += [("subsampling/conv0", nn.Conv2d(1, 32, 3, (stride[0], 1))), ("subsampling/relu0", nn.ReLU())]
        for i in range(1, num_layers):
            layers += [(f"subsampling/pad{i}", nn.ConstantPad2d((0, 0, 2, 0), 0))]
            layers += [(f"subsampling/conv{i}", nn.Conv2d(32, 32, 3, (stride[i], 1))), (f"subsampling/relu{i}", nn.ReLU())]
        layers = OrderedDict(layers)
        self.conv = nn.Sequential(layers)
        self.affine = nn.Linear(32 * (idim - 2 * num_layers), odim)
        self.norm = nn.LayerNorm(odim)

    def forward(self, feats, feat_lengths=None):
        outputs = feats.unsqueeze(1)  # [T, C, B, D]
        outputs = outputs.permute(2, 1, 0, 3)  # [B, C, T, D]
        outputs = self.conv(outputs)
        outputs = outputs.permute(2, 0, 1, 3).contiguous()

        T, B, C, D = outputs.size()
        outputs = self.affine(outputs.view(T, B, C * D))

        outputs = self.norm(outputs)

        if feat_lengths is not None:
            feat_lengths = torch.as_tensor(feat_lengths)
            for i in range(self.num_layers):
                feat_lengths = (feat_lengths - 1) // self.stride[i] + 1

        return outputs, feat_lengths


class PositionalEncoding(nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param int max_len: maximum input length
    :param reverse: whether to reverse the input position

    """

    def __init__(self, d_model, max_len=2000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.reverse = reverse
        self.scale = math.sqrt(self.d_model)
        self.pe = None

        self._extend_pe(torch.tensor(0.0).expand(max_len, 1))

    def _extend_pe(self, x):
        """Reset the positional encodings."""
        T = x.size(0)
        if self.pe is None or self.pe.size(0) < T:
            pe = torch.zeros(T, self.d_model)
            if self.reverse:
                position = torch.arange(T - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
            else:
                position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)

            div_term = torch.exp( torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(1)

        self.pe = self.pe.to(x)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (time, batch, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (time, batch, ...)

        """
        self._extend_pe(x)
        outputs = self.scale * x + self.pe[:x.size(0), :]
        return outputs


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, layer_drop=0.0, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.layer_drop = layer_drop

    def reset_parameters(self, layer_index_offset=0):
        for layer_idx, enc_layer in enumerate(self.layers):
            enc_layer.reset_parameters(layer_index=layer_idx + layer_index_offset + 1)

    def forward(self, src: Tensor, layer: Optional[int] = None, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) \
                -> Tuple[Tensor, Optional[List[Tensor]]]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for idx, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            if ((idx + 1) == layer):
                return output

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VGGTFEncoder(nn.Module):

    def __init__(self,
                 input_size,
                 nhead,
                 d_model,
                 dim_feedforward,
                 num_encoder_layers,
                 dropout=0.0,
                 layer_drop=0.0,
                 activation="gelu",
                 subsampling="conv2d",
                 conv2d_stride=None,
                 num_conv_layers=2):
        super(VGGTFEncoder, self).__init__()

        self.subsampling = Conv2dSubsampling(input_size,
                                             d_model,
                                             num_layers=num_conv_layers,
                                             stride=conv2d_stride)
        self.pe = PositionalEncoding(d_model)
        self.pe_dropout = nn.Dropout(dropout)
        self.conv2d_stride = conv2d_stride
        self.num_encoder_layers = num_encoder_layers
        encoder_norm = None

        # FB type
        encoder_layer = TransformerPreLNEncoderLayer(d_model,
                                                     nhead,
                                                     dim_feedforward,
                                                     dropout,
                                                     final_norm=True)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          layer_drop, encoder_norm)

    def forward(self, x, batch_sizes=None):
        x, batch_sizes = self.subsampling(x.transpose(0, 1), batch_sizes)
        x = self.pe_dropout(self.pe(x))
        key_padding_mask = generate_padding_mask(x, batch_sizes)
        x = self.encoder(x, mask=None, src_key_padding_mask=key_padding_mask)

        return x, batch_sizes

    def get_mid_emb(self, x, batch_sizes=None, layer=None):
        x, batch_sizes = self.subsampling(x.transpose(0, 1), batch_sizes)
        x = self.pe_dropout(self.pe(x))
        key_padding_mask = generate_padding_mask(x, batch_sizes)
        x = self.encoder(x, layer=layer, mask=None, src_key_padding_mask=key_padding_mask)

        return x, batch_sizes


def build_VGGtf_encoder(model_cfg):
    feat_dim = model_cfg.get("feat_dim")
    delta_order = model_cfg.get("delta_order")
    left = model_cfg.get("left")
    right = model_cfg.get("right")
    input_size = (delta_order + 1) * (1 + left + right) * feat_dim

    nhead = model_cfg.get("nhead")
    d_model = model_cfg.get("d_model")
    dim_feedforward = model_cfg.get("dim_feedforward")
    num_encoder_layers = model_cfg.get("num_encoder_layers")
    dropout = model_cfg.get("dropout")
    layer_drop = model_cfg.get("layer_drop")
    activation = model_cfg.get("activation")

    subsampling = model_cfg.get("subsampling")
    num_conv_layers = model_cfg.get("num_conv_layers")
    conv2d_stride = model_cfg.get("conv2d_stride")

    model = VGGTFEncoder(input_size,
                         nhead,
                         d_model,
                         dim_feedforward,
                         num_encoder_layers,
                         dropout=dropout,
                         layer_drop=layer_drop,
                         activation=activation,
                         subsampling=subsampling,
                         conv2d_stride=conv2d_stride,
                         num_conv_layers=num_conv_layers)

    return model


def build_model(config):
    encoder_type = config["type"]
    model_cfg = config[encoder_type]
    if encoder_type == "VGGtf_encoder":
        return build_VGGtf_encoder(model_cfg)
    else:
        assert False, "invalid encoder_type"


class Kmeans(nn.Module):

    def __init__(self,
                 codebook_dim=1280,
                 codebook_size=8192):
        super(Kmeans, self).__init__()

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        codebook = torch.normal(0.0, (1 / self.codebook_size**0.5), size=(self.codebook_dim, self.codebook_size))
        self.register_buffer("codebook", codebook)

    @torch.no_grad()
    def forward(self, x):
        # input: x: (..., D)
        # codebook (D, C)
        ori_size = x.size()

        x = x.view(-1, x.size(-1))
        x = torch.nn.functional.normalize(x, dim=-1)

        d1 = torch.sum(x**2, dim=-1, keepdim=True)
        d2 = torch.sum(self.codebook**2, dim=0)
        d3 = torch.matmul(x, self.codebook)
        d = d1 + d2 - 2 * d3

        inds = d.argmin(-1)
        inds = inds.view(ori_size[:-1])

        return inds, self.codebook[:,inds]
    
    def get_codebook(self):
        return self.codebook


encoder_config_path_prefix = os.path.dirname(__file__) + "/configs"


def select_config(semantic_tokenizer_type="vgg_asr_60ms_layer26_codebook8192"):
    encoder_config_name = f"{semantic_tokenizer_type}.yaml"
    encoder_config_path = os.path.join(encoder_config_path_prefix, encoder_config_name)

    assert os.path.exists(encoder_config_path), f"config path {encoder_config_path} is not exist!"

    with open(encoder_config_path) as f:
        config = yaml.safe_load(f.read())
        semantic_tokenizer_config = config['semantic_tokenizer_config']

    return semantic_tokenizer_config


class WavToLabel(nn.Module):

    def __init__(self, semantic_tokenizer_type="Long"):
        super(WavToLabel, self).__init__()

        semantic_tokenizer_config = select_config(semantic_tokenizer_type)
        select_layer = semantic_tokenizer_config['select_layer']
        codebook_dim = semantic_tokenizer_config['codebook_dim']
        codebook_size = semantic_tokenizer_config['codebook_size']
        model_config = semantic_tokenizer_config['model_config']
        sr = semantic_tokenizer_config['sample_rate']

        # Fix path for our structure
        if not os.path.isabs(model_config):
            # If relative path, make it relative to this module's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Try direct path first
            test_path = os.path.join(current_dir, 'configs', 'semantic_tokenizer.infer.yaml')
            if os.path.exists(test_path):
                model_config = test_path
            else:
                # Fallback: try from project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
                model_config = os.path.join(project_root, model_config)
                model_config = os.path.normpath(model_config)

        with open(model_config) as f:
            config = yaml.safe_load(f.read())

        self.feature_extractor = FeatureExtractor(config["feature"])
        self.model = build_model(config['encoder'])

        self.kmeans = Kmeans(codebook_size=codebook_size, codebook_dim=codebook_dim)

        self.layer = select_layer
        self.sr = sr

    def set_sr(self, sr):
        self.sr = sr

    @torch.no_grad()
    def forward(self, wavs, wav_lens):
        # wavs: [B, T], wav_lens: [B]
        wavs = wavs.float()
        feats, feat_lens = self.feature_extractor(wavs, wav_lens)
        # feats, feat_lens = self.feature(wavs, wav_lens)
        feats, feat_lens = self.model.get_mid_emb(feats, feat_lens, self.layer)
        feats = feats.transpose(0, 1).contiguous()
        kmeans_labels, kmeans_feats = self.kmeans(feats)

        return kmeans_labels, feat_lens, feats, kmeans_feats

    def get_codebook(self):
        return self.kmeans.get_codebook()