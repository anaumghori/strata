from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Block:
    """Represents a single block in the paged KV cache"""
    block_id: int
    ref_count: int = 0
    hash_value: int = -1
    token_ids: list[int] = field(default_factory=list)


    def reset(self) -> None:
        """Reset block to initial state for reuse"""
        self.ref_count = 1
        self.hash_value = -1
        self.token_ids = []


class BlockManager:
    """
    Manages paged block allocation for the KV cache. Handles allocation, deallocation, 
    and tracking of free and used blocks across all sequences.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        """Initialize block manager with specified capacity.

        :param num_blocks: Total number of blocks available
        :param block_size: Tokens per block
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(block_id=i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()


    @property
    def num_free_blocks(self) -> int:
        """Returns: Number of free blocks available"""
        return len(self.free_block_ids)


    def can_allocate(self, num_blocks_needed: int) -> bool:
        """Check if the requested number of blocks can be allocated.
        
        :param num_blocks_needed: Number of blocks required
        :returns: True if allocation is possible
        """
        return self.num_free_blocks >= num_blocks_needed


    def allocate_blocks(self, num_blocks: int) -> list[int]:
        """Allocate the specified number of blocks.

        :param num_blocks: Number of blocks to allocate
        :returns: List of allocated block IDs
        """
        allocated = []
        for _ in range(num_blocks):
            if not self.free_block_ids:
                break
            block_id = self.free_block_ids.popleft()
            self.blocks[block_id].reset()
            self.used_block_ids.add(block_id)
            allocated.append(block_id)
        return allocated


    def allocate_single_block(self) -> Optional[int]:
        """Allocate a single block if available.

        :returns: Block ID or None if no blocks available
        """
        if not self.free_block_ids:
            return None
        block_id = self.free_block_ids.popleft()
        self.blocks[block_id].reset()
        self.used_block_ids.add(block_id)
        return block_id


    def deallocate_blocks(self, block_ids: list[int]) -> None:
        """Deallocate the specified blocks.

        :param block_ids: List of block IDs to deallocate
        :returns: None
        """
        for block_id in reversed(block_ids):
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count <= 0:
                    self.used_block_ids.discard(block_id)
                    self.free_block_ids.append(block_id)


    def get_num_blocks_for_tokens(self, num_tokens: int) -> int:
        """
        :param num_tokens: Number of given tokens to store
        :returns: Number of blocks required
        """
        return (num_tokens + self.block_size - 1) // self.block_size


    def can_append_token(self, block_table: list[int], current_length: int) -> bool:
        """Check if a token can be appended to the sequence.

        :param block_table: Current block table for the sequence
        :param current_length: Current sequence length
        :returns: True if token can be appended
        """
        offset_in_block = current_length % self.block_size
        if offset_in_block == 0:
            return self.num_free_blocks >= 1
        return True


    def maybe_allocate_for_append(self, block_table: list[int], current_length: int) -> list[int]:
        """Allocate new blocks if needed to cover all positions up to current_length.

        :param block_table: Current block table for the sequence
        :param current_length: Current sequence length
        :returns: Updated block table
        """
        blocks_needed = self.get_num_blocks_for_tokens(current_length)
        while len(block_table) < blocks_needed:
            new_block_id = self.allocate_single_block()
            if new_block_id is not None:
                block_table.append(new_block_id)
            else:
                break
        return block_table


    def get_slot_mapping(self, block_table: list[int], start_pos: int, end_pos: int) -> list[int]:
        """Compute slot mapping for a range of positions.

        :param block_table: Block table for the sequence
        :param start_pos: Start position (inclusive)
        :param end_pos: End position (exclusive)
        :returns: List of slot indices
        """
        slot_mapping = []
        for pos in range(start_pos, end_pos):
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            if block_idx < len(block_table):
                block_id = block_table[block_idx]
                slot = block_id * self.block_size + offset
                slot_mapping.append(slot)
        return slot_mapping
