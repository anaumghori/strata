from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import xxhash
import numpy as np


@dataclass
class Block:
    """
    Represents a single block in the paged KV cache. Tracks reference count for sharing, hash 
    value for prefix caching, and token IDs for hash verification and cache management.
    """

    block_id: int
    ref_count: int = 0
    hash_value: int = -1
    token_ids: list[int] = field(default_factory=list)


    def reset(self) -> None:
        """Reset block to initial state for fresh allocation."""
        self.ref_count = 1
        self.hash_value = -1
        self.token_ids = []


    def set_cached(self, hash_value: int, token_ids: list[int]) -> None:
        """Mark block as cached with computed hash and tokens.

        :param hash_value: Computed hash for this block
        :param token_ids: Token IDs stored in this block
        """
        self.hash_value = hash_value
        self.token_ids = token_ids


    @property
    def is_cached(self) -> bool:
        """Check if block has valid cached content."""
        return self.hash_value != -1


class BlockManager:
    """
    Manages paged block allocation with prefix caching support. Uses hash-based lookup to identify and 
    reuse blocks containing identical token sequences, enabling computation reuse for shared prefixes.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = True,
    ) -> None:
        """
        :param num_blocks: Total number of blocks available
        :param block_size: Tokens per block
        :param enable_prefix_caching: Enable hash-based prefix caching
        """

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [Block(block_id=i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.hash_to_block_id: dict[int, int] = {}


    @staticmethod
    def compute_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
        """
        :param token_ids: Token IDs in this block
        :param prefix_hash: Hash of preceding blocks (-1 if first block)
        :returns: Computed hash value
        """
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()


    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)


    def can_allocate(self, num_blocks_needed: int) -> bool:
        """
        :param num_blocks_needed: Number of blocks required
        :returns: True if allocation is possible
        """
        return self.num_free_blocks >= num_blocks_needed


    def _allocate_block_by_id(self, block_id: int) -> Block:
        """
        :param block_id: Specific block ID to allocate
        :returns: Allocated Block object
        """
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self.free_block_ids.remove(block_id)
        block.ref_count += 1
        self.used_block_ids.add(block_id)
        return block


    def allocate_blocks(self, num_blocks: int) -> list[int]:
        """Allocate the specified number of fresh blocks.

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
        """Allocate a single fresh block if available.

        :returns: Block ID or None if no blocks available
        """
        if not self.free_block_ids:
            return None
        block_id = self.free_block_ids.popleft()
        self.blocks[block_id].reset()
        self.used_block_ids.add(block_id)
        return block_id


    def allocate_with_prefix_cache(self, prompt_tokens: list[int]) -> tuple[list[int], int]:
        """Allocate blocks for a prompt, reusing cached prefix blocks.

        :param prompt_tokens: Full prompt token sequence
        :returns: Tuple of (block_ids, num_cached_tokens)
        """
        if not self.enable_prefix_caching:
            num_blocks = self.get_num_blocks_for_tokens(len(prompt_tokens))
            return self.allocate_blocks(num_blocks), 0

        num_blocks = self.get_num_blocks_for_tokens(len(prompt_tokens))
        block_ids = []
        num_cached_tokens = 0
        prefix_hash = -1
        cache_miss = False

        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(prompt_tokens))
            block_tokens = prompt_tokens[start:end]
            is_full_block = len(block_tokens) == self.block_size
            if is_full_block and not cache_miss:
                block_hash = self.compute_hash(block_tokens, prefix_hash)
                cached_block_id = self.hash_to_block_id.get(block_hash, -1)

                if cached_block_id != -1:
                    cached_block = self.blocks[cached_block_id]
                    if cached_block.token_ids == block_tokens:
                        self._allocate_block_by_id(cached_block_id)
                        block_ids.append(cached_block_id)
                        num_cached_tokens += self.block_size
                        prefix_hash = block_hash
                        continue

                cache_miss = True
            new_block_id = self.allocate_single_block()
            if new_block_id is None:
                break

            if is_full_block:
                block_hash = self.compute_hash(block_tokens, prefix_hash)
                block = self.blocks[new_block_id]
                block.set_cached(block_hash, block_tokens)
                self.hash_to_block_id[block_hash] = new_block_id
                prefix_hash = block_hash
            block_ids.append(new_block_id)
        return block_ids, num_cached_tokens


    def cache_completed_block(
        self,
        block_id: int,
        token_ids: list[int],
        prefix_hash: int,
    ) -> int:
        """Register a completed block in the prefix cache.

        :param block_id: Block to cache
        :param token_ids: Tokens in the block
        :param prefix_hash: Hash of preceding blocks
        :returns: Computed hash for this block
        """
        if not self.enable_prefix_caching:
            return -1

        block_hash = self.compute_hash(token_ids, prefix_hash)
        block = self.blocks[block_id]
        block.set_cached(block_hash, token_ids)
        self.hash_to_block_id[block_hash] = block_id
        return block_hash


    def deallocate_blocks(self, block_ids: list[int]) -> None:
        """Param: block_ids: List of block IDs to deallocate"""
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
        """
        :param block_table: Current block table for the sequence
        :param current_length: Current sequence length
        :returns: True if token can be appended to the sequence.
        """
        offset_in_block = current_length % self.block_size
        if offset_in_block == 0:
            return self.num_free_blocks >= 1
        return True


    def maybe_allocate_for_append(self, block_table: list[int], current_length: int) -> list[int]:
        """Allocate new blocks if needed to cover all positions.

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


    def get_slot_mapping(
        self,
        block_table: list[int],
        start_pos: int,
        end_pos: int,
    ) -> list[int]:
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


    def get_block_hash(self, block_id: int) -> int:
        return self.blocks[block_id].hash_value # -1 if not cached


    @property
    def num_cached_blocks(self) -> int:
        """Returns: Number of blocks with valid cache entries."""
        return len(self.hash_to_block_id)
