import torch
from torch import Tensor

from strata.cache.kv_cache import KVCache
from strata.cache.conv_state import ConvStateStore
from strata.cache.block_manager import BlockManager


class StateManager:
    """
    Unified interface for managing all model state. Coordinates KV cache and conv 
    state allocation, providing a single point of access for the model runner.
    """

    def __init__(
        self,
        num_attention_layers: int,
        num_shortconv_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_sequences: int,
        conv_kernel_size: int,
        conv_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize state manager with all cache components.

        :param num_attention_layers: Number of attention layers
        :param num_shortconv_layers: Number of ShortConv layers
        :param num_blocks: Total KV cache blocks
        :param block_size: Tokens per block
        :param num_kv_heads: Key-value heads per attention layer
        :param head_dim: Dimension per head
        :param max_sequences: Maximum concurrent sequences
        :param conv_kernel_size: ShortConv kernel size
        :param conv_dim: ShortConv hidden dimension
        :param dtype: Data type for cache tensors
        """
        self.block_size = block_size
        self.max_sequences = max_sequences
        self.dtype = dtype

        self.kv_cache = KVCache(
            num_layers=num_attention_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

        self.conv_state = ConvStateStore(
            num_layers=num_shortconv_layers,
            max_sequences=max_sequences,
            kernel_size=conv_kernel_size,
            conv_dim=conv_dim,
            dtype=dtype,
        )

        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )

        self.seq_to_slot: dict[int, int] = {}
        self.seq_to_blocks: dict[int, list[int]] = {}


    def allocate_sequence(self, seq_id: int, initial_length: int) -> bool:
        """Allocate cache resources for a new sequence.

        :param seq_id: Unique sequence identifier
        :param initial_length: Initial prompt length
        :returns: True if allocation succeeded
        """
        num_blocks_needed = self.block_manager.get_num_blocks_for_tokens(initial_length)
        if not self.block_manager.can_allocate(num_blocks_needed):
            return False
        block_ids = self.block_manager.allocate_blocks(num_blocks_needed)
        self.seq_to_blocks[seq_id] = block_ids
        slot = self.conv_state.allocate_slot(seq_id)
        self.seq_to_slot[seq_id] = slot
        return True


    def deallocate_sequence(self, seq_id: int) -> None:
        """Deallocate all cache resources for a sequence.

        :param seq_id: Sequence identifier to deallocate
        """
        if seq_id in self.seq_to_blocks:
            block_ids = self.seq_to_blocks.pop(seq_id)
            self.block_manager.deallocate_blocks(block_ids)

        if seq_id in self.seq_to_slot:
            slot = self.seq_to_slot.pop(seq_id)
            self.conv_state.deallocate_slot(slot)


    def can_allocate(self, num_tokens: int) -> bool:
        """Check if resources can be allocated for given token count.

        :param num_tokens: Number of tokens to check
        :returns: True if allocation is possible
        """
        num_blocks = self.block_manager.get_num_blocks_for_tokens(num_tokens)
        return self.block_manager.can_allocate(num_blocks)


    def can_append_token(self, seq_id: int, current_length: int) -> bool:
        """Check if a token can be appended to the sequence.

        :param seq_id: Sequence identifier
        :param current_length: Current sequence length
        :returns: True if append is possible
        """
        if seq_id not in self.seq_to_blocks:
            return False
        block_table = self.seq_to_blocks[seq_id]
        return self.block_manager.can_append_token(block_table, current_length)


    def allocate_for_append(self, seq_id: int, current_length: int) -> None:
        """Allocate additional block if needed for append.

        :param seq_id: Sequence identifier
        :param current_length: Current sequence length
        """
        if seq_id in self.seq_to_blocks:
            block_table = self.seq_to_blocks[seq_id]
            self.block_manager.maybe_allocate_for_append(block_table, current_length)


    def get_block_table(self, seq_id: int) -> list[int]:
        """Get the block table for a sequence.

        :param seq_id: Sequence identifier
        :returns: List of block IDs
        """
        return self.seq_to_blocks.get(seq_id, [])


    def get_conv_slot(self, seq_id: int) -> int:
        """Get the conv state slot for a sequence.

        :param seq_id: Sequence identifier
        :returns: Slot index
        """
        return self.seq_to_slot.get(seq_id, 0)


    def get_slot_mapping(
        self,
        seq_id: int,
        start_pos: int,
        end_pos: int,
    ) -> list[int]:
        """Compute slot mapping for cache writes.

        :param seq_id: Sequence identifier
        :param start_pos: Start position
        :param end_pos: End position
        :returns: List of slot indices
        """
        block_table = self.get_block_table(seq_id)
        return self.block_manager.get_slot_mapping(block_table, start_pos, end_pos)


    def get_kv_caches(self) -> list[Tensor]:
        """Returns: List of KV cache tensors per layer for model forward"""
        return self.kv_cache.get_all_caches()


    def get_conv_states(self) -> list[Tensor]:
        """Returns: List of conv state tensors per layer for model forward"""
        return [
            self.conv_state.get_full_state_for_layer(i)
            for i in range(self.conv_state.num_layers)
        ]


    def update_conv_state(self, layer_idx: int, seq_ids: list[int], new_state: Tensor) -> None:
        """Update conv state after prefill.

        :param layer_idx: ShortConv layer index
        :param seq_ids: List of sequence IDs
        :param new_state: New state values per sequence
        :returns: None
        """
        slots = torch.tensor(
            [self.get_conv_slot(sid) for sid in seq_ids],
            dtype=torch.long,
        )
        self.conv_state.set_state(layer_idx, slots, new_state)


    @property
    def num_free_blocks(self) -> int:
        """Returns: Number of free blocks available"""
        return self.block_manager.num_free_blocks


    def memory_usage_bytes(self) -> int:
        """Calculate total memory usage of all caches"""
        return self.kv_cache.memory_usage_bytes() + self.conv_state.memory_usage_bytes()
