from typing import Optional
import torch
from torch import nn, Tensor

from strata.models.base import BaseModelForCausalLM
from strata.models.lfm.config import (
    LfmConfig,
    load_lfm_config,
    get_attention_layer_indices,
    get_shortconv_layer_indices,
    _is_attention_layer,
)
from strata.models.lfm.layers import RMSNorm, Lfm2DecoderLayer
from strata.models.lfm.weights import load_lfm_weights


class LfmForCausalLM(BaseModelForCausalLM):
    
    def __init__(
        self,
        config: LfmConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize LFM model with all layers.

        :param config: LFM model configuration
        :param dtype: Data type for model weights
        """
        super().__init__()
        self.config = config
        self.dtype = dtype

        self._attention_layer_indices = get_attention_layer_indices(config)
        self._shortconv_layer_indices = get_shortconv_layer_indices(config)

        self._attn_idx_to_layer_idx = {
            layer_idx: attn_idx
            for attn_idx, layer_idx in enumerate(self._attention_layer_indices)
        }
        self._shortconv_idx_to_layer_idx = {
            layer_idx: sc_idx
            for sc_idx, layer_idx in enumerate(self._shortconv_layer_indices)
        }

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
        )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            is_attention = _is_attention_layer(config.layer_types[i])
            layer = Lfm2DecoderLayer(config, i, is_attention, dtype=dtype)
            self.layers.append(layer)

        self.embedding_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, dtype=dtype
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
        )


    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: Optional[list[Tensor]] = None,
        slot_mapping: Optional[Tensor] = None,
        context_lens: Optional[Tensor] = None,
        block_tables: Optional[Tensor] = None,
        seq_lens: Optional[list[int]] = None,
        is_prefill: bool = True,
        conv_states: Optional[Tensor] = None,
        seq_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[list[Tensor]]]:
        """Execute forward pass through the complete model.

        :param input_ids: Token IDs tensor of shape [num_tokens]
        :param positions: Position indices tensor of shape [num_tokens]
        :param kv_caches: List of KV cache tensors per attention layer
        :param slot_mapping: Mapping of tokens to cache slots
        :param context_lens: Context lengths per sequence for decode
        :param block_tables: Block table mapping for paged attention
        :param seq_lens: Sequence lengths for prefill attention masking
        :param is_prefill: Whether this is a prefill or decode step
        :param conv_states: Conv state tensor for ShortConv layers
        :param seq_ids: Sequence IDs for conv state indexing
        :returns: Tuple of (logits, new_conv_states per ShortConv layer)
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        new_conv_states = []
        attn_idx = 0
        shortconv_idx = 0

        for i, layer in enumerate(self.layers):
            is_attention = layer.is_attention

            layer_kv_cache = None
            layer_conv_state = None

            if is_attention and kv_caches is not None:
                layer_kv_cache = kv_caches[attn_idx]

            if not is_attention and conv_states is not None:
                layer_conv_state = conv_states[shortconv_idx]

            hidden_states, residual, new_conv_state = layer(
                hidden_states,
                positions,
                residual,
                kv_cache=layer_kv_cache,
                conv_state=layer_conv_state,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_table=block_tables,
                seq_ids=seq_ids,
                seq_lens=seq_lens,
                is_prefill=is_prefill,
                attn_layer_idx=attn_idx if is_attention else 0,
                shortconv_layer_idx=shortconv_idx if not is_attention else 0,
            )

            if is_attention:
                attn_idx += 1
            else:
                if new_conv_state is not None:
                    new_conv_states.append(new_conv_state)
                shortconv_idx += 1

        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        logits = self.lm_head(hidden_states)

        return logits, new_conv_states if new_conv_states else None


    def load_weights(self, model_path: str) -> None:
        load_lfm_weights(self, model_path, self.config)


    def tie_weights(self) -> None:
        """Tie embedding and lm_head weights if configured"""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight


    @property
    def num_attention_layers(self) -> int:
        """Return the number of attention layers requiring KV cache"""
        return len(self._attention_layer_indices)


    @property
    def num_shortconv_layers(self) -> int:
        """Return the number of ShortConv layers requiring conv state"""
        return len(self._shortconv_layer_indices)


    @property
    def num_kv_heads(self) -> int:
        """Return the number of key-value heads per attention layer"""
        return self.config.num_key_value_heads


    @property
    def head_dim(self) -> int:
        """Return the dimension per attention head"""
        return self.config.head_dim


    @property
    def conv_dim(self) -> int:
        """Return the dimension of ShortConv layers"""
        return self.config.conv_dim


    @property
    def conv_kernel_size(self) -> int:
        """Return the kernel size for ShortConv layers"""
        return self.config.conv_kernel_size


def build_lfm_model(
    model_path: str,
    dtype: torch.dtype = torch.float32,
) -> LfmForCausalLM:
    """
    :param model_path: Path to model directory
    :param dtype: Data type for model weights
    :returns: Initialized and loaded LFM model
    """
    config = load_lfm_config(model_path)
    model = LfmForCausalLM(config, dtype=dtype)
    model.load_weights(model_path)
    model.tie_weights()
    model.eval()
    return model
