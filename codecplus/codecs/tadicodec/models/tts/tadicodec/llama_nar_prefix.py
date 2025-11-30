from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import torch
import torch.nn.functional as F
import numpy as np
import os
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
        # Calculate the embedding frequencies
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
        self,
        hidden_size: int = 1024,
        eps: float = 1e-6,
        dim_cond: int = 1024,
        use_cond: bool = True,
    ):
        """
        Initializes the LlamaAdaptiveRMSNorm module.

        Args:
            hidden_size (int): The dimension of the hidden states to be normalized.
            eps (float): A small value added to the variance for numerical stability.
            dim_cond (int): The dimension of the conditional embedding.
            use_cond (bool): If True, use conditional embedding to modulate the output.
        """
        super().__init__()
        self.use_cond = use_cond
        if self.use_cond:
            # Linear layer to project the conditional embedding to the hidden size
            self.to_weight = nn.Linear(dim_cond, hidden_size)
            # Initialize weights to zero and bias to one for an identity transformation at the start
            nn.init.zeros_(self.to_weight.weight)
            nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True  # Disable automatic Hugging Face initialization

    def forward(
        self, hidden_states: torch.Tensor, cond_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies the adaptive RMS normalization.

        Args:
            hidden_states (torch.Tensor): The input tensor, shape `(batch, seq_len, hidden_size)`.
            cond_embedding (Optional[torch.Tensor]): The conditional embedding, shape `(batch, dim_cond)` or `(batch, seq_len, dim_cond)`.

        Returns:
            torch.Tensor: The normalized and modulated hidden states.
        """
        input_dtype = hidden_states.dtype
        # Calculate variance and normalize the hidden states
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Apply conditional modulation if enabled
        if self.use_cond:
            # Project conditional embedding to get the scaling weight
            weight = self.to_weight(cond_embedding)
            # Unsqueeze if the conditional embedding is per-batch instead of per-token
            if len(weight.shape) == 2:
                weight = weight.unsqueeze(1)

            # Apply the learned scaling factor
            return (weight * hidden_states).to(input_dtype)
        else:
            return hidden_states


class LlamaNARDecoderLayer(LlamaDecoderLayer):
    """
    A Non-Autoregressive (NAR) Llama Decoder Layer using adaptive layer normalization.

    This class overrides the standard LlamaDecoderLayer to replace its RMSNorm
    modules with LlamaAdaptiveRMSNorm, allowing it to be conditioned on an external embedding
    (e.g., from diffusion timesteps).
    """

    def __init__(self, config: LlamaConfig, layer_idx: int, use_cond: bool = True):
        """
        Overrides the LlamaDecoderLayer to use adaptive layer normalization.

        Args:
            config (LlamaConfig): The configuration object for the Llama model.
            layer_idx (int): The index of the layer.
            use_cond (bool): Whether to use adaptive conditioning in the layer norms.
        """
        super().__init__(config, layer_idx)  # init attention, mlp, etc.
        # Override the standard layer norms with our adaptive versions
        self.input_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dim_cond=config.hidden_size,
            use_cond=use_cond,
        )
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dim_cond=config.hidden_size,
            use_cond=use_cond,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Forward pass for the NAR decoder layer, including conditional embedding.

        Args:
            hidden_states (torch.FloatTensor): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            cond_embedding (torch.Tensor): Conditional embedding for adaptive normalization.
            attention_mask (Optional[torch.Tensor]): Attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (Optional[torch.LongTensor]): Indices of positions of each input sequence tokens.
            past_key_value (Optional[Tuple[torch.Tensor]]): Cached past key and value projection states.
            output_attentions (Optional[bool]): Whether or not to return the attentions tensors of all attention layers.
            use_cache (Optional[bool]): If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.

        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
            A tuple containing the output hidden states and optionally the attention weights and past key/value states.
        """
        residual = hidden_states

        # Apply adaptive pre-attention layer norm
        hidden_states = self.input_layernorm(
            hidden_states, cond_embedding=cond_embedding
        )

        # Self Attention block
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        # Handle both old (3 values) and new (2 values) transformers API
        if len(attn_output) == 3:
            hidden_states, self_attn_weights, present_key_value = attn_output
        else:
            hidden_states, present_key_value = attn_output
            self_attn_weights = None
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
    A Llama-based non-autoregressive transformer model for diffusion tasks.

    This model uses a Llama architecture but modifies it for non-autoregressive generation.
    Key features:
    1.  Non-causal (fully-visible) attention mask.
    2.  Adaptive layer normalization conditioned on diffusion timesteps.
    3.  Ability to be conditioned on text embeddings, which are prepended as a prefix.
    4.  Input and output linear projection layers for feature mapping.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 16,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        use_text_emb: bool = True,
        use_diff_step: bool = True,
        use_cond: bool = True,
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
            in_dim (Optional[int]): Dimension of the input features. If set, an input linear layer is created.
            out_dim (Optional[int]): Dimension of the output features. If set, an output linear layer is created.
            use_text_emb (bool): Whether to use text embeddings as a conditional prefix.
            use_diff_step (bool): Whether to use diffusion timestep conditioning.
            use_cond (bool): Whether to use an additional per-token conditional input `cond`.
            config (LlamaConfig): A LlamaConfig object. A default is provided for convenience.
        """
        super().__init__(config)

        self.use_text_emb = use_text_emb
        self.use_diff_step = use_diff_step
        self.use_cond = use_cond
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Create a stack of non-autoregressive Llama layers
        layer_config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            max_position_embeddings=4096,
            intermediate_size=hidden_size * 4,
            _attn_implementation="eager",  # Use eager attention for compatibility
        )
        self.layers = nn.ModuleList(
            [
                LlamaNARDecoderLayer(
                    layer_config,
                    layer_idx=i,
                    use_cond=use_cond,
                )
                for i in range(num_layers)
            ]
        )

        # Final adaptive layer norm
        self.norm = LlamaAdaptiveRMSNorm(
            hidden_size,
            dim_cond=hidden_size,
            use_cond=(use_cond or use_diff_step),
        )

        # MLP for processing text embedding condition
        if self.use_text_emb:
            self.text_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        # Modules for processing diffusion timestep condition
        if self.use_diff_step:
            self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
            self.diff_step_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        # MLP for processing the general per-token `cond` condition
        if self.use_cond:
            self.cond_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        # Ensure all layers use the correct adaptive norm configuration
        for layer in self.layers:
            layer.input_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size,
                dim_cond=hidden_size,
                use_cond=(use_cond or use_diff_step),
            )
            layer.post_attention_layernorm = LlamaAdaptiveRMSNorm(
                hidden_size,
                dim_cond=hidden_size,
                use_cond=(use_cond or use_diff_step),
            )

        # We handle embeddings manually, so disable the default token embedder
        self.embed_tokens = None

        # Initialize rotary embeddings for position encoding (needed for transformers >= 4.46.0)
        self.rotary_emb = LlamaRotaryEmbedding(config=layer_config)

        # Optional input and output projection layers
        if self.in_dim is not None:
            self.in_linear = nn.Linear(self.in_dim, hidden_size)
        if self.out_dim is not None:
            self.out_linear = nn.Linear(hidden_size, self.out_dim)

        self.post_init()

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> Optional[torch.Tensor]:
        """
        Creates a non-causal (fully-visible) attention mask.

        This method overrides the default causal mask creation in LlamaModel.
        It converts a 2D padding mask `[bsz, seq_len]` into a 4D attention mask
        `[bsz, 1, tgt_seq_len, src_seq_len]` suitable for self-attention,
        without applying a causal triangle.

        Args:
            attention_mask (torch.Tensor): The 2D padding mask.
            input_shape (Tuple[int, int, int]): The shape of the input embeddings.
            inputs_embeds (torch.Tensor): The input embeddings tensor.
            past_key_values_length (int): The length of any cached key-values.

        Returns:
            Optional[torch.Tensor]: The 4D attention mask, or None if the input mask is None.
        """
        combined_attention_mask = None

        def _expand_mask(
            mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
        ) -> torch.Tensor:
            """
            Expands a 2D attention mask to a 4D attention mask.
            """
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len

            expanded_mask = (
                mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            )

            # Invert the mask: 1.0 for valid tokens, 0.0 for padded tokens
            # and then convert to additive mask format (-inf for masked positions)
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
        x_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        diffusion_step: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass of the DiffLlamaPrefix model.

        Args:
            x (torch.Tensor): The primary input tensor, e.g., noisy data in diffusion `(batch, seq_len, in_dim)`.
            x_mask (torch.Tensor): The padding mask for `x`, shape `(batch, seq_len)`.
            cond (Optional[torch.Tensor]): Additional per-token conditional input, shape `(batch, seq_len, hidden_size)`.
            diffusion_step (Optional[torch.Tensor]): Diffusion timesteps, shape `(batch,)`.
            text_embedding (Optional[torch.Tensor]): Text embeddings to be used as a prefix, shape `(batch, text_len, hidden_size)`.
            text_mask (Optional[torch.Tensor]): The padding mask for `text_embedding`, shape `(batch, text_len)`.
            input_ids, attention_mask, etc.: Standard Hugging Face arguments (mostly unused here but kept for compatibility).

        Returns:
            Union[torch.Tensor, Tuple]: The final output tensor `(batch, seq_len, out_dim)`.
            If `output_hidden_states` is True, returns a tuple `(output_tensor, all_hidden_states)`.
        """
        # 1. Input and Conditional Projections
        if self.in_dim is not None:
            x = self.in_linear(x)

        if self.use_cond and cond is not None:
            cond_embedding = self.cond_mlp(cond)  # (B, T, C)
            x = x + cond_embedding

        # 2. Prepend Text Embedding Prefix
        if self.use_text_emb and text_embedding is not None:
            text_embedding = self.text_mlp(text_embedding)  # (B, T, C)
            text_length = text_embedding.shape[1]
            # Concatenate text prefix and main input
            inputs_embeds = torch.cat([text_embedding, x], dim=1)
            attention_mask = torch.cat([text_mask, x_mask], dim=1)
        else:
            inputs_embeds = x
            attention_mask = x_mask
            text_length = 0

        # 3. Process Diffusion Step Embedding for Adaptive Norm
        if self.use_diff_step and diffusion_step is not None:
            # Convert scalar timesteps to vector embeddings and project with an MLP
            diffusion_step_emb = self.diff_step_embedding(diffusion_step).to(x.device)
            diffusion_step_emb = self.diff_step_mlp(diffusion_step_emb)  # (B, C)
        else:
            diffusion_step_emb = None

        # 4. Standard Transformer Preamble (adapted from LlamaModel)
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

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        # Create the non-causal attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        # Compute position embeddings (for transformers >= 4.46.0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 5. Transformer Decoder Layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                # Store hidden states before the layer, excluding the text prefix part
                all_hidden_states += (hidden_states[:, text_length:],)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # Pass the processed diffusion step embedding to the adaptive layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cond_embedding=diffusion_step_emb,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 6. Final Normalization and Output Processing
        hidden_states = self.norm(hidden_states, cond_embedding=diffusion_step_emb)

        if output_hidden_states:
            # Add hidden states from the last decoder layer
            all_hidden_states += (hidden_states[:, text_length:],)

        next_cache = next_decoder_cache if use_cache else None

        # Remove the text prefix from the final output sequence
        hidden_states = hidden_states[
            :,
            text_length:,
        ]

        # Apply final output projection if specified
        if self.out_dim is not None:
            hidden_states = self.out_linear(hidden_states)

        # 7. Return results (simplified for this snippet)
        if output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
