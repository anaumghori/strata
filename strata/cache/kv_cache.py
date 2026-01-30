import torch
from torch import Tensor


class KVCache:
    """
    Paged key-value cache for attention layers. Manages block-based storage of K and V tensors across 
    all attention layers, supporting efficient allocation and deallocation for variable-length sequences.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize paged KV cache with pre-allocated blocks.

        :param num_layers: Number of attention layers
        :param num_blocks: Total number of cache blocks
        :param block_size: Tokens per block
        :param num_kv_heads: Number of key-value heads
        :param head_dim: Dimension per head
        :param dtype: Data type for cache tensors
        """
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.k_caches = []
        self.v_caches = []

        for _ in range(num_layers):
            k_cache = torch.zeros(
                num_blocks, block_size, num_kv_heads, head_dim,
                dtype=dtype,
            )
            v_cache = torch.zeros(
                num_blocks, block_size, num_kv_heads, head_dim,
                dtype=dtype,
            )
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)


    def get_layer_cache(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Get the KV cache tensors for a specific layer.

        :param layer_idx: Index of the attention layer
        :returns: Tuple of (k_cache, v_cache) tensors
        """
        return (self.k_caches[layer_idx], self.v_caches[layer_idx])


    def get_all_caches(self) -> list[Tensor]:
        """Returns: List of stacked KV tensors per layer"""
        return [self.get_layer_cache(i) for i in range(self.num_layers)]


    def memory_usage_bytes(self) -> int:
        """Calculate total memory usage of the KV cache"""
        element_size = self.k_caches[0].element_size()
        elements_per_layer = self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim
        return 2 * self.num_layers * elements_per_layer * element_size
