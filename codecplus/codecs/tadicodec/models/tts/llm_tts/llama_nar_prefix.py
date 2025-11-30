from transformers import LlamaConfig, LlamaModel
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import math

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding module.

    This module generates sinusoidal positional embeddings for a given 1D input tensor,
    which is commonly used for representing timesteps in diffusion models.
    """

    def __init__(self, dim: int):
        """
        Initializes the SinusoidalPosEmb module.

        Args:
            dim (int): The dimension of the embedding.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates the positional embedding.

        Args:
            x (torch.Tensor): A 1D tensor of positions (e.g., timesteps), shape `(batch_size,)`.

        Returns:
            torch.Tensor: The positional embeddings, shape `(batch_size, dim)`.
        """
        device = x.device
        half_dim = self.dim // 2
        # Calculate the embedding frequencies based on the log-space formula
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Create the embedding matrix by multiplying positions with frequencies
        emb = x[:, None] * emb[None, :] * 1.0
        # Concatenate sine and cosine components to form the final embedding
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    """
    Adaptive Root Mean Square Layer Normalization.

    This is a variant of RMSNorm where the scaling factor (weight) is adaptively
    predicted from a conditional embedding, allowing for conditional modulation
    of the normalized hidden states.
    """

    def __init__(
        self, hidden_size: int = 1024, eps: float = 1e-6, dim_cond: int = 1024
    ):
        """
        Initializes the LlamaAdaptiveRMSNorm module.

        Args:
            hidden_size (int): The dimension of the hidden states to be normalized.
            eps (float): A small value added to the variance for numerical stability.
            dim_cond (int): The dimension of the conditional embedding.
        """
        super().__init__()
        # Linear layer to project the conditional embedding to the hidden size
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        # Initialize weights to zero and bias to one for an identity transformation at the start
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        # Disable automatic Hugging Face initialization for this custom module
        self._is_hf_initialized = True

    def forward(
        self, hidden_states: torch.Tensor, cond_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the adaptive RMS normalization.

        Args:
            hidden_states (torch.Tensor): The input tensor, shape `(batch, seq_len, hidden_size)`.
            cond_embedding (torch.Tensor): The conditional embedding, shape `(batch, dim_cond)`.

        Returns:
            torch.Tensor: The normalized and modulated hidden states.
        """
        input_dtype = hidden_states.dtype
        # Calculate variance and normalize the hidden states
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Predict the scaling factor from the conditional embedding
        weight = self.to_weight(cond_embedding)
        # Unsqueeze if the conditional embedding is per-batch instead of per-token
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        # Apply the learned scaling factor
        return (weight * hidden_states).to(input_dtype)


class LlamaNARDecoderLayer(LlamaDecoderLayer):
    """
    A Non-Autoregressive (NAR) Llama Decoder Layer using adaptive layer normalization.

    This class overrides the standard LlamaDecoderLayer to replace its RMSNorm
    modules with LlamaAdaptiveRMSNorm, allowing it to be conditioned on an external embedding.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        """Overrides the LlamaDecoderLayer to use adaptive layer normalization."""
        super().__init__(config, layer_idx)  # init attention, mlp, etc. from parent
        self.layer_idx = layer_idx
        # Override the standard layer norms with our adaptive versions
        self.input_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Forward pass for the NAR decoder layer, including conditional embedding.

        Args:
            hidden_states (torch.Tensor): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            cond_embedding (torch.Tensor): Conditional embedding for adaptive normalization.
            attention_mask (Optional[torch.Tensor]): Attention mask of size `(batch, 1, tgt_len, src_len)`.
            position_ids (Optional[torch.LongTensor]): Indices of positions of each input sequence tokens.
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Cached past key and value projection states.
            output_attentions (Optional[bool]): Whether to return the attention tensors.
            use_cache (Optional[bool]): If True, past key values are returned to speed up decoding.

        Returns:
            Tuple containing the output hidden states, and optionally attention weights and past key/value states.
        """
        residual = hidden_states

        # Apply adaptive pre-attention layer norm
        hidden_states = self.input_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )

        # Self Attention block
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected block
        residual = hidden_states
        # Apply adaptive post-attention layer norm
        hidden_states = self.post_attention_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class DiffLlamaPrefix(LlamaModel):
    """
    A Llama-based non-autoregressive transformer model for diffusion (masked generative modeling) tasks.

    This model uses a Llama architecture but modifies it for non-autoregressive generation.
    Key features:
    1. Non-causal (fully-visible) attention mask.
    2. Adaptive layer normalization conditioned on diffusion timesteps.
    3. Ability to be conditioned on phoneme (text) embeddings, which are prepended as a prefix.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 16,
        use_phone_cond: bool = True,
        config: LlamaConfig = LlamaConfig(
            vocab_size=0, hidden_size=256, num_attention_heads=1, num_hidden_layers=1
        ),
    ):
        """
        Initializes the DiffLlamaPrefix model.

        Args:
            hidden_size (int): The hidden dimension of the transformer.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of transformer layers.
            use_phone_cond (bool): Whether to use phoneme embeddings as a conditional prefix.
            config (LlamaConfig): A LlamaConfig object. A default is provided for convenience.
        """
        super().__init__(config)

        self.use_phone_cond = use_phone_cond

        # Create a stack of non-autoregressive Llama layers
        self.layers = nn.ModuleList(
            [
                LlamaNARDecoderLayer(
                    LlamaConfig(
                        hidden_size=hidden_size,
                        num_attention_heads=num_heads,
                        max_position_embeddings=4096,
                        intermediate_size=hidden_size * 4,
                    ),
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )

        # Final adaptive layer norm
        self.norm = LlamaAdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        # Modules for diffusion step conditioning
        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # MLP for processing phoneme embedding condition
        if self.use_phone_cond:
            self.cond_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        # This loop is redundant if layers are initialized correctly, but ensures config consistency.
        for layer in self.layers:
            layer.input_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )
            layer.post_attention_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size, dim_cond=hidden_size
            )

        # We handle embeddings manually, so disable the default token embedder
        self.embed_tokens = None

        self.post_init()

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> Optional[torch.Tensor]:
        """
        Creates a non-causal (fully-visible) attention mask.

        This method overrides the default causal mask creation. It converts a 2D padding mask
        `[bsz, seq_len]` into a 4D attention mask `[bsz, 1, tgt_seq_len, src_seq_len]`
        suitable for self-attention, without applying a causal triangle.

        Args:
            attention_mask (torch.Tensor): The 2D padding mask.
            input_shape (Tuple[int, int]): The shape of the input (`batch_size`, `seq_len`).
            inputs_embeds (torch.Tensor): The input embeddings tensor.
            past_key_values_length (int): The length of any cached key-values.

        Returns:
            Optional[torch.Tensor]: The 4D attention mask, or None if the input mask is None.
        """
        combined_attention_mask = None

        def _expand_mask(
            mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
        ) -> torch.Tensor:
            """Expands a 2D attention mask to a 4D attention mask."""
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len
            expanded_mask = (
                mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            )
            # Invert the mask and convert to additive format (-inf for masked positions)
            inverted_mask = 1.0 - expanded_mask
            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        x: torch.Tensor,
        diffusion_step: torch.Tensor,
        x_mask: torch.Tensor,
        phone_embedding: Optional[torch.Tensor] = None,
        phone_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the DiffLlamaPrefix model.

        Args:
            x (torch.Tensor): The primary input tensor, e.g., noisy data (`batch, seq_len, hidden_size`).
            diffusion_step (torch.Tensor): Diffusion timesteps, shape `(batch,)`.
            x_mask (torch.Tensor): The padding mask for `x`, shape `(batch, seq_len)`.
            phone_embedding (Optional[torch.Tensor]): Phoneme embeddings prefix, shape `(batch, phone_len, hidden_size)`.
            phone_mask (Optional[torch.Tensor]): The padding mask for `phone_embedding`, shape `(batch, phone_len)`.
            input_ids, etc.: Standard Hugging Face arguments, mostly for compatibility.

        Returns:
            torch.Tensor: The final output tensor of shape `(batch, seq_len, hidden_size)`.
        """
        # 1. Prepend conditional prefix (phoneme embeddings)
        if self.use_phone_cond and phone_embedding is not None:
            # Process condition through an MLP
            phone_embedding = self.cond_mlp(phone_embedding)  # (B, T_phone, C)
            phone_length = phone_embedding.shape[1]
            # Concatenate prefix and main input
            inputs_embeds = torch.cat([phone_embedding, x], dim=1)
            attention_mask = torch.cat([phone_mask, x_mask], dim=1)
        else:
            inputs_embeds = x
            attention_mask = x_mask
            phone_length = 0

        # 2. Process diffusion step embedding for adaptive normalization
        diffusion_step_emb = self.diff_step_embedding(diffusion_step).to(x.device)
        diffusion_step_emb = self.diff_step_mlp(diffusion_step_emb)  # (B, C)

        # 3. Standard Transformer Preamble (adapted from LlamaModel)
        batch_size, seq_length, _ = inputs_embeds.shape

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        # Create the non-causal attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        # 4. Transformer Decoder Layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # Pass the processed diffusion step embedding to the adaptive layer
            layer_outputs = decoder_layer(
                hidden_states,
                cond_embedding=diffusion_step_emb,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 5. Final Normalization and Output Processing
        hidden_states = self.norm(hidden_states, cond_embedding=diffusion_step_emb)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Remove the conditional prefix from the final output sequence
        return hidden_states[:, phone_length:]
