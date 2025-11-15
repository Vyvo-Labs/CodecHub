import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_bigcodec import BigCodecConfig

# 请确保这些模块路径是正确的
from .vq.codec_encoder import CodecEncoder_Transformer
from .vq.codec_decoder_vocos import CodecDecoderVocos
from .vq.module import SemanticEncoder
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torch.nn.functional as F
class XCodec2Model(PreTrainedModel):
    config_class = BigCodecConfig

    def __init__(self, config: BigCodecConfig):
        super().__init__(config)

        # 1) 语义模型
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0",
            output_hidden_states=True
        )
        self.semantic_model.eval()

        self.SemanticEncoder_module = SemanticEncoder(
             config.semantic_hidden_size,
             config.semantic_hidden_size,
             config.semantic_hidden_size
        )

        # 2) Codec Encoder
        self.CodecEnc = CodecEncoder_Transformer()

        # 3) Codec Decoder
        self.generator = CodecDecoderVocos(
            hidden_dim=config.codec_decoder_hidden_size,
            hop_length=config.hop_length,
            sample_rate=config.sample_rate,
            upsample_factors=config.upsample_factors,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
        )

        # 4) 两个全连接层
        self.fc_prior = nn.Linear(2048, 2048)
        self.fc_post_a = nn.Linear(2048, 1024)
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.feature_extractor = feature_extractor

    def forward(self, input_waveform, sample_rate=16000):
        """
        这里的 forward 不一定要叫 forward，也可以拆成别的方法；
        但是如果想兼容 pipeline，需要在 forward 里给出核心逻辑。

        参数：
          input_waveform: [batch_size, waveform_length]
          sample_rate: 默认 16000
        返回：
          重构后的语音音频 (Tensor)
        """
        # 1) 特征提取
        # 如果需要 padding，可以在这里做
        wav = input_waveform
        pad_for_wav = (320 - (wav.shape[1] % 320))

        wav = torch.nn.functional.pad(wav, (0, pad_for_wav))

        input_features = self.feature_extractor(
            F.pad(wav[0,:].cpu(), (160, 160)),
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)  # [batch, frames, feat_dim]

        # 2) 语义层
        semantic_output = self.semantic_model(input_features)
        semantic_hidden_16 = semantic_output.hidden_states[16]  # 取第16层
        semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
        semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

        # 3) codec encoder
        wav = wav.to(self.device)  # shape: [batch, 1, time]
        vq_emb = self.CodecEnc(wav.unsqueeze(1))  # [batch, time//down, 1024] 只是示例
        vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]



        # 4) 拼接
        concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 1024 + 1024, frames]

        # 5) fc_prior
        concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        # 6) decoder 的量化部分
        _, vq_code, _ = self.generator(concat_emb, vq=True)
        vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
        vq_post_emb = vq_post_emb.transpose(1, 2)

        # 7) fc_post_a
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)

        # 8) 最后解码成波形
        recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]
        # recon_audio: [batch, time]
        return recon_audio

    def encode_code(self, input_waveform, sample_rate=16000):
        """
        将输入的音频编码为代码表示。

        参数：
          input_waveform: [batch_size, waveform_length]
          sample_rate: 默认 16000
        返回：
          编码后的代码 (Tensor)
        """
        with torch.no_grad():


            wav = input_waveform
            pad_for_wav = (320 - (wav.shape[1] % 320))

            wav = torch.nn.functional.pad(wav, (0, pad_for_wav))

            input_features = self.feature_extractor(
                F.pad(wav[0,:].cpu(), (160, 160)),
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features.to(self.device)  # [batch, frames, feat_dim]

            # 2) 语义层
            semantic_output = self.semantic_model(input_features)
            semantic_hidden_16 = semantic_output.hidden_states[16]  # 取第16层
            semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
            semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

            # 3) codec encoder
            wav = wav.to(self.device)  # shape: [batch, 1, time]
            vq_emb = self.CodecEnc(wav.unsqueeze(1))  # [batch, time//down, 1024] 只是示例
            vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

            # 4) 拼接
            concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 2048, frames]

            # 5) fc_prior
            concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

            # 6) decoder 的量化部分，获取code
            _, vq_code, _ = self.generator(concat_emb, vq=True)
            # vq_code: [batch, frames]
            return vq_code






    def decode_code(self, vq_code):
        """
        将编码后的代码解码回音频。

        参数：
          vq_code: 编码后的代码 (Tensor) [batch, frames]
        返回：
          解码后的音频 (Tensor) [batch, waveform_length]
        """
        with torch.no_grad():
            # 获取量化后的嵌入
            vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
            vq_post_emb = vq_post_emb.transpose(1, 2)  # [batch, 1024, frames]

            # 7) fc_post_a
            vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)  # [batch, 1024, frames]

            # 8) 最后解码成波形
            recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]  # [batch, time]
            return recon_audio


    def encode_batch_feats(self, input_waveform, input_features):
        """
        将输入的音频编码为代码表示。

        参数：
          input_waveform: [batch_size,1, waveform_length]
          input_features:
        返回：
          编码后的代码 (Tensor)
        """
        with torch.no_grad():

            # 2) 语义层
            semantic_output = self.semantic_model(input_features[:,0,:,:])
            semantic_hidden_16 = semantic_output.hidden_states[16]  # 取第16层
            semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
            semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

            # 3) codec encoder
            wav = input_waveform  #.unsqueeze(1).to(self.device)  # shape: [batch, 1, time]
            vq_emb = self.CodecEnc(wav)  # [batch, time//down, 1024] 只是示例
            vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]


            # 4) 拼接
            concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 2048, frames]

            # 5) fc_prior
            concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

            # 6) decoder 的量化部分，获取code
            _, vq_code, _ = self.generator(concat_emb, vq=True)
            # vq_code: [batch, frames]
            return vq_code
