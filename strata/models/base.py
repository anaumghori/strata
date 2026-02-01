from abc import ABC, abstractmethod
from typing import Optional
from torch import nn, Tensor


class BaseModelForCausalLM(ABC, nn.Module):
    """
    Abstract base class for language models in Strata. Defines the interface that all 
    model implementations must follow.
    """

    @abstractmethod
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
    ) -> Tensor:
        """Execute forward pass through the model.

        :param input_ids: Token IDs tensor of shape [num_tokens]
        :param positions: Position indices tensor of shape [num_tokens]
        :param kv_caches: List of KV cache tensors per attention layer
        :param slot_mapping: Mapping of tokens to cache slots
        :param context_lens: Context lengths per sequence for decode
        :param block_tables: Block table mapping for paged attention
        :param seq_lens: Sequence lengths for prefill attention masking
        :param is_prefill: Whether this is a prefill or decode step
        :returns: Logits tensor of shape [num_tokens, vocab_size]
        """
        pass


    @abstractmethod
    def load_weights(self, model_path: str) -> None:
        """model_path: Path to directory containing weight files"""
        pass


    @property
    @abstractmethod
    def num_attention_layers(self) -> int:
        """Return the number of attention layers requiring KV cache"""
        pass


    @property
    @abstractmethod
    def num_shortconv_layers(self) -> int:
        """Return the number of ShortConv layers requiring conv state in the model"""
        pass


    @property
    @abstractmethod
    def num_kv_heads(self) -> int:
        """Return the number of key-value heads per attention layer"""
        pass


    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Return the dimension per attention head"""
        pass


    @property
    @abstractmethod
    def conv_dim(self) -> int:
        """Return the dimension of ShortConv layers"""
        pass


    @property
    @abstractmethod
    def conv_kernel_size(self) -> int:
        """Return the kernel size for ShortConv layers"""
        pass
