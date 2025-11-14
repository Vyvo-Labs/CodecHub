import math
from typing import List, Union, Tuple

import numpy as np

import torch
from torch import nn
import torchaudio

from codecplus.codecs.longcat.networks.quantizers.auto_grvq import AutoGroupResidualVectorQuantize
from codecplus.codecs.longcat.networks.modules.model import Encoder
from codecplus.codecs.longcat.networks.decoders.mem_decoders import MemDecoder
from codecplus.codecs.longcat.semantic_tokenizer_general.encoder import WavToLabel

def init_weights(m):
    """Initializes weights for convolutional layers."""
    if isinstance(m, nn.Conv1d):
        # nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.orthogonal_(m.weight)  # Use orthogonal initialization to prevent exploding gradients.
        # nn.init.constant_(m.bias, 0)


class SingleTokenDequantizer(nn.Module):
    """A simple module to dequantize integer tokens into continuous embeddings."""

    def __init__(self, n_vocabulary: int, codebook_dim: int):
        super().__init__()

        self.dequant = nn.Embedding(n_vocabulary, codebook_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor of token indices into their corresponding embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            A tensor of integer token indices.
            
        Returns
        -------
        torch.Tensor
            The continuous embedding vectors for the input tokens.
        """
        return self.dequant(x)


class LongCatAudioCodecEncoder(nn.Module):
    """
    LongCatAudioCodecEncoder.

    This module acts as a dual-path "tokenizer", processing an input audio waveform
    to extract two parallel streams of discrete integer codes:
    1.  Semantic Tokens: Extracted using a pre-trained, frozen semantic tokenizer,
        representing the content of the speech.
    2.  Acoustic Tokens: Learned through a convolutional encoder and quantized
        using residual vector quantization (RVQ). This acoustic path is not
        trained as a standalone codec; instead, it is co-trained to act as a
        supplement to the semantic tokens, capturing residual information such as
        speaker identity, prosody, and other acoustic details necessary for
        high-fidelity reconstruction.

    The primary output is a list containing these two sets of discrete tokens.
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        input_sample_rate: int = 16000,
        semantic_tokenizer_type: str = 'LongCat_semantic_tokenizer',
    ):
        """
        Initializes the LongCatAudioCodec_encoder.

        Parameters
        ----------
        encoder_dim : int, optional
            Base dimension for the acoustic convolutional encoder.
        encoder_rates : List[int], optional
            A list of downsampling rates for each layer of the acoustic encoder.
        latent_dim : int, optional
            The dimension of the latent space produced by the acoustic encoder.
            If None, it is calculated as `encoder_dim * (2**len(encoder_rates))`.
        n_codebooks : int, optional
            Number of codebooks for the acoustic residual vector quantizer.
        codebook_size : int, optional
            The number of entries (vectors) in each acoustic codebook.
        codebook_dim : Union[int, list], optional
            The dimension of the vectors in the acoustic codebook.
        input_sample_rate : int, optional
            The expected sample rate of the input audio in Hz.
        semantic_tokenizer_type : str, optional
            The identifier for the pre-trained semantic tokenizer model to be loaded.
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.input_sample_rate = input_sample_rate
        if latent_dim is None:
            latent_dim = encoder_dim * (2**len(encoder_rates))

        self.latent_dim = latent_dim
        
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.hop_length = np.prod(encoder_rates)

        self.n_codebooks = n_codebooks
        self.acoustic_codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.acoustic_quantizer = AutoGroupResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            frame_residual_vq=False
        )
        if self.acoustic_codebook_size == 0:
            latent_dim = 0 # If codebook_size is 0, the acoustic path provides no information
        self.apply(init_weights)

        self.valid_semantic_tokenizer_types = {
            "LongCatAudioCodec_semantic_tokenizer",
        }

        assert semantic_tokenizer_type in self.valid_semantic_tokenizer_types, \
            f"Semantic tokenizer type '{semantic_tokenizer_type}' is not supported."

        print(f'Begin loading semantic tokenizer: {semantic_tokenizer_type}.')

        # This part loads a pre-trained model for semantic tokenization.
        self.semantic_tokenizer = WavToLabel(semantic_tokenizer_type=semantic_tokenizer_type).eval()
        print('Finished loading semantic tokenizer.')

    def preprocess(self, audio_data: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Prepares the audio data by resampling (if needed) and padding.

        Parameters
        ----------
        audio_data : torch.Tensor
            The input audio waveform.
        sample_rate : int
            The sample rate of the input `audio_data`. If `None`, it is assumed
            to match `self.input_sample_rate`.

        Returns
        -------
        torch.Tensor
            The preprocessed audio data.
        """
        if sample_rate is None:
            sample_rate = self.input_sample_rate
        
        if sample_rate != self.input_sample_rate:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=sample_rate, new_freq=self.input_sample_rate)
        
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def get_semantic_codes(self, audio_data: torch.Tensor) -> torch.Tensor:
        """
        Extracts discrete semantic codes from audio using the pre-trained tokenizer.

        Parameters
        ----------
        audio_data : torch.Tensor, shape [B, 1, T_audio]
            The input audio waveform.

        Returns
        -------
        torch.Tensor
            Discrete semantic token indices of shape [B, T_codes].
        """
        batch_size = audio_data.shape[0]
        length_padding = audio_data.shape[-1]
        
        with torch.no_grad():
            wav_lens = torch.IntTensor([length_padding] * batch_size).to(audio_data.device)
            # The semantic tokenizer expects audio in the int16 range.
            codes, _, _, _ = self.semantic_tokenizer(audio_data.squeeze(1) * 32767, wav_lens)
            semantic_codes = codes.clone().detach()

        return semantic_codes
    
    def get_semantic_codes_with_lengths(self, audio_data: torch.Tensor, wav_lens: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts discrete semantic codes from a batch of variable-length audio.

        Parameters
        ----------
        audio_data : torch.Tensor, shape [B, 1, T_audio]
            Padded audio data for the batch.
        wav_lens : torch.Tensor, optional
            A tensor of original audio waveform lengths.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - semantic_codes (torch.Tensor, shape [B, T_codes]): Discrete semantic token indices.
            - codes_lens (torch.Tensor): The valid lengths of each semantic code sequence.
        """
        batch_size = audio_data.shape[0]

        with torch.no_grad():
            if wav_lens is None:
                wav_lens = torch.IntTensor([audio_data.shape[-1]] * batch_size).to(audio_data.device)
            # The semantic tokenizer expects audio in the int16 range.
            codes, codes_lens, _, _ = self.semantic_tokenizer(audio_data.squeeze(1) * 32767, wav_lens)
            semantic_codes = codes.clone().detach()
   
        return semantic_codes, codes_lens

    def get_acoustic_codes(self, audio_data: torch.Tensor, n_acoustic_codebooks: int = None) -> torch.Tensor:
        """
        Generates discrete acoustic codes from audio (acoustic path only).

        Parameters
        ----------
        audio_data : torch.Tensor, shape [B, 1, T_audio]
            Audio data to encode.
        n_acoustic_codebooks : int, optional
            Number of acoustic quantizers to use. If None, all available quantizers are used.

        Returns
        -------
        torch.Tensor
            Discrete acoustic codebook indices of shape [B, N_q, T_z].
        """
        if n_acoustic_codebooks is None:
            # If no specific number is requested, use all available codebooks.
            acoustic_code_book_num = self.n_codebooks
        else:
            # If a specific number is requested, validate it first.
            assert 0 < n_acoustic_codebooks <= self.n_codebooks, \
                f"Requested n_acoustic_codebooks ({n_acoustic_codebooks}) is out of the valid range (1 to {self.n_codebooks})."

            acoustic_code_book_num = n_acoustic_codebooks

        z = self.encoder(audio_data)
        _, codes, _, _, _ = self.acoustic_quantizer(z, acoustic_code_book_num)

        return codes

    def get_acoustic_codes_with_lengths(self, audio_data: torch.Tensor, audio_lens: torch.Tensor = None, n_acoustic_codebooks: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates discrete acoustic codes from a batch of variable-length audio.

        Parameters
        ----------
        audio_data : torch.Tensor, shape [B, 1, T_audio]
            Padded audio data for the batch.
        audio_lens : torch.Tensor, optional
            A tensor containing the original lengths of each audio in the batch.
        n_acoustic_codebooks : int, optional
            Number of acoustic quantizers to use. If None, all are used.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - codes (torch.Tensor, shape [B, N_q, T_z]): Discrete codebook indices.
            - z_lens (torch.Tensor): The valid lengths of each sequence in the encoded output.
        """
        if n_acoustic_codebooks is None:
            # If no specific number is requested, use all available codebooks.
            acoustic_code_book_num = self.n_codebooks
        else:
            # If a specific number is requested, validate it first.
            assert 0 < n_acoustic_codebooks <= self.n_codebooks, \
                f"Requested n_acoustic_codebooks ({n_acoustic_codebooks}) is out of the valid range (1 to {self.n_codebooks})."

            acoustic_code_book_num = n_acoustic_codebooks

        if audio_lens is not None:
            z, z_lens = self.encoder.forward_with_lens(audio_data, batch_sizes=audio_lens)
        else:
            z = self.encoder(audio_data)
            z_lens = None
            
        _, codes, _, _, _ = self.acoustic_quantizer(z, acoustic_code_book_num)

        return codes, z_lens

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_acoustic_codebooks: int = None,
    ) -> List[torch.Tensor]:
        """
        Main forward pass to tokenize audio into semantic and acoustic codes.

        Parameters
        ----------
        audio_data : torch.Tensor, shape [B, 1, T_audio]
            Audio data to encode.
        sample_rate : int, optional
            The sample rate of the input `audio_data`. If `None`, it is assumed
            to match `self.input_sample_rate`.
        n_acoustic_codebooks : int, optional
            Number of acoustic quantizers to use. If None, all are used.

        Returns
        -------
        List[torch.Tensor]
            A list containing the two token streams: `[semantic_codes, acoustic_codes]`.
        """
        assert audio_data.dim() == 3 and audio_data.shape[1] == 1, \
            "Input audio_data must be a 3D tensor of shape [B, 1, T]."

        if n_acoustic_codebooks is None:
            # If no specific number is requested, use all available codebooks.
            acoustic_code_book_num = self.n_codebooks
        else:
            # If a specific number is requested, validate it first.
            assert 0 < n_acoustic_codebooks <= self.n_codebooks, \
                f"Requested n_acoustic_codebooks ({n_acoustic_codebooks}) is out of the valid range (1 to {self.n_codebooks})."

            acoustic_code_book_num = n_acoustic_codebooks
            
        # Preprocess audio: resample if necessary and pad to a valid length.
        audio_data = self.preprocess(audio_data, sample_rate)
        
        # Get semantic tokens.
        semantic_codes = self.get_semantic_codes(audio_data)

        # Get acoustic tokens if the acoustic path is enabled.
        if self.acoustic_codebook_size != 0:
            acoustic_codes = self.get_acoustic_codes(audio_data, acoustic_code_book_num)
        else:
            acoustic_codes = None
            
        return [semantic_codes, acoustic_codes]
    
    def get_semantic_codebook(self) -> torch.Tensor:
        """
        Retrieves the codebook from the semantic tokenizer.

        Returns
        -------
        torch.Tensor
            The semantic codebook embeddings.
        """
        return self.semantic_tokenizer.get_codebook()


class LongCatAudioCodecDecoder(nn.Module):
    """
    LongCatAudioCodecDecoder.

    This module reconstructs an audio waveform from discrete semantic and acoustic tokens.
    It contains its own dequantization layers to convert the input integer tokens
    back into continuous latent representations before feeding them to the waveform generator.
    """
    def __init__(
        self,
        latent_dim: int = 1024,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        semantic_dim: int = 1280,
        decoder_type: str = '16k',
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        semantic_token_nums: int = 8192,
    ):
        """
        Initializes the LongCatAudioCodec_decoder.

        Parameters
        ----------
        latent_dim : int, optional
            The dimension of the acoustic portion of the latent space.
        decoder_dim : int, optional
            The base channel dimension for the decoder's convolutional layers.
        decoder_rates : List[int], optional
            A list of upsampling rates for each layer of the decoder.
        semantic_dim : int, optional
            The dimension of the semantic portion of the latent space.
        decoder_type : str, optional
            Specifies the decoder architecture ('16k' or '24k').
        n_codebooks : int, optional
            Number of codebooks for the internal acoustic dequantizer.
        codebook_size : int, optional
            The number of entries in each acoustic codebook.
        codebook_dim : Union[int, list], optional
            The dimension of the vectors in the acoustic codebook.
        semantic_token_nums : int, optional
            The vocabulary size for the internal semantic dequantizer.
        """
        super().__init__()

        assert decoder_type in ['16k', '24k'], f"Unsupported decoder_type: '{decoder_type}'. Must be '16k' or '24k'."

        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.decoder_type = decoder_type
        self.acoustic_codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        
        if decoder_type == '16k':
            self.output_rate = 16000
        else: # '24k'
            self.output_rate = 24000
        
        total_input_dim = latent_dim + semantic_dim
        
        # Initialize the main waveform generator (e.g., GAN generator).
        if self.decoder_type == '16k':
            self.decoder = MemDecoder(
                total_input_dim, decoder_dim, decoder_rates, groups=1,
                lookahead_frame=3, lstm_nums=1, is_final_causal=False,
            )
        else: # '24k'
            self.decoder = MemDecoder(
                total_input_dim, decoder_dim, decoder_rates, groups=1,
                lookahead_frame=3, lstm_nums=1, is_final_causal=True,
            )
            
        # Initialize internal dequantization modules.
        self.acoustic_quantizer = AutoGroupResidualVectorQuantize(
            input_dim=latent_dim, n_codebooks=n_codebooks, codebook_size=codebook_size,
            codebook_dim=codebook_dim, frame_residual_vq=False
        )

        self.semantic_dequantizer = SingleTokenDequantizer(
            n_vocabulary=semantic_token_nums,
            codebook_dim=semantic_dim
        )
        
        self.apply(init_weights)


    def forward(self, semantic_codes: torch.Tensor, acoustic_codes: torch.Tensor) -> torch.Tensor:
        """
        Decodes semantic and acoustic tokens into an audio waveform.

        This method first dequantizes the input tokens to get continuous latents,
        concatenates them, and then feeds the result to a waveform generator.

        Parameters
        ----------
        semantic_codes : torch.Tensor, shape [B, T_codes]
            The discrete semantic tokens.
        acoustic_codes : torch.Tensor, shape [B, N_q, T_codes]
            The discrete acoustic tokens from the residual vector quantizer.

        Returns
        -------
        torch.Tensor
            The reconstructed audio waveform of shape [B, 1, T_audio].
        """
        # Dequantize semantic tokens to get continuous latent features.
        semantic_latents = self.semantic_dequantizer(semantic_codes)
        
        if self.acoustic_codebook_size != 0 and acoustic_codes is not None:
            # Dequantize acoustic tokens using the internal RVQ's codebooks.
            assert 0 < acoustic_codes.shape[1] <= self.n_codebooks, \
                   f"The acoustic codes is encoded in {acoustic_codes.shape[1]} acoustic codebook, but the decoder only supply at most {self.n_codebooks} acoustic codebook"
            acoustic_latents, _, _ = self.acoustic_quantizer.from_codes(acoustic_codes)
            
            # Align features by truncating to the minimum length before concatenation.
            min_len = min(semantic_latents.shape[1], acoustic_latents.shape[-1])
            latents = torch.cat((semantic_latents.transpose(-1, -2)[..., :min_len], acoustic_latents[..., :min_len]), dim=1)
        else:
            # If no acoustic path, use only semantic latents.
            latents = semantic_latents.transpose(-1, -2)
            
        # Generate waveform from the combined latent representation.
        quantized_audio = self.decoder(latents)

        return quantized_audio


    def forward_stream(self, semantic_codes: torch.Tensor, acoustic_codes: torch.Tensor, list1_h_input: torch.Tensor, list1_c_input: torch.Tensor, list2_h_input: torch.Tensor, list2_c_input: torch.Tensor, lstm_buffer_input: torch.Tensor) -> torch.Tensor:
        """
        Decodes semantic and acoustic tokens into an audio waveform in streaming mode.

        This method first dequantizes the input tokens to get continuous latents,
        concatenates them, and then feeds the result to a waveform generator in
        streaming fashion with LSTM memory states.

        Parameters
        ----------
        semantic_codes : torch.Tensor, shape [B, T_codes]
            The discrete semantic tokens.
        acoustic_codes : torch.Tensor, shape [B, N_q, T_codes]
            The discrete acoustic tokens from the residual vector quantizer.
        list1_h_input : torch.Tensor
            Previous hidden state for first LSTM memory layer
        list1_c_input : torch.Tensor  
            Previous cell state for first LSTM memory layer
        list2_h_input : torch.Tensor
            Previous hidden state for second LSTM memory layer
        list2_c_input : torch.Tensor
            Previous cell state for second LSTM memory layer
        lstm_buffer_input : torch.Tensor
            LSTM buffer input for maintaining streaming context

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - quantized_audio_slice: Generated audio slice of shape [B, 1, T_slice]
            - list1_h: Updated hidden state for first LSTM memory layer
            - list1_c: Updated cell state for first LSTM memory layer
            - list2_h: Updated hidden state for second LSTM memory layer
            - list2_c: Updated cell state for second LSTM memory layer
            - lstm_buffer: Updated LSTM buffer for next iteration
        """
        # Dequantize semantic tokens to get continuous latent features.
        semantic_latents = self.semantic_dequantizer(semantic_codes)
        
        if self.acoustic_codebook_size != 0 and acoustic_codes is not None:
            # Dequantize acoustic tokens using the internal RVQ's codebooks.
            assert 0 < acoustic_codes.shape[1] <= self.n_codebooks, \
                   f"The acoustic codes is encoded in {acoustic_codes.shape[1]} acoustic codebook, but the decoder only supply at most {self.n_codebooks} acoustic codebook"
            acoustic_latents, _, _ = self.acoustic_quantizer.from_codes(acoustic_codes)
            
            # Align features by truncating to the minimum length before concatenation.
            min_len = min(semantic_latents.shape[1], acoustic_latents.shape[-1])
            latents = torch.cat((semantic_latents.transpose(-1, -2)[..., :min_len], acoustic_latents[..., :min_len]), dim=1)
        else:
            # If no acoustic path, use only semantic latents.
            latents = semantic_latents.transpose(-1, -2)
            
        # Generate waveform from the combined latent representation in streaming mode
        quantized_audio_slice, list1_h, list1_c, list2_h, list2_c, lstm_buffer = self.decoder.forward_stream(
            latents, list1_h_input, list1_c_input, list2_h_input, list2_c_input, lstm_buffer_input
        )
        
        return quantized_audio_slice, list1_h, list1_c, list2_h, list2_c, lstm_buffer