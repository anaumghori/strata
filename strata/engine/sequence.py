from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Optional

from strata.config import SamplingParams


class SequenceStatus(Enum):
    """
    Status of a sequence in the inference pipeline. Tracks whether a sequence is waiting for 
    prefill, running decode, or has completed generation.
    """

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    Tracks state for a single inference request. Manages token IDs, prefill progress, cache handles,
    and generation parameters throughout the sequence lifecycle.
    """

    _counter = count()

    def __init__(
        self,
        prompt_tokens: list[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> None:
        """Initialize a sequence with prompt tokens and sampling config.

        :param prompt_tokens: List of input token IDs
        :param sampling_params: Generation parameters
        """
        self.seq_id = next(Sequence._counter)
        self.prompt_tokens = copy(prompt_tokens)
        self.generated_tokens: list[int] = []
        self.status = SequenceStatus.WAITING
        self.num_prefilled_tokens = 0
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.conv_state_slot: int = -1

        if sampling_params is None:
            sampling_params = SamplingParams()

        self.max_new_tokens = sampling_params.max_tokens
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.stop_sequences = sampling_params.stop_sequences
        self.ignore_eos = sampling_params.ignore_eos


    @property
    def prompt_length(self) -> int:
        """Returns: Prompt token count"""
        return len(self.prompt_tokens)


    @property
    def generated_length(self) -> int:
        """Returns: Generated token count"""
        return len(self.generated_tokens)


    @property
    def current_length(self) -> int:
        """Returns: Total token count (prompt + generated)"""
        return self.prompt_length + self.generated_length


    @property
    def last_token(self) -> int:
        """Returns: Last token ID (most recent token in the sequence)"""
        if self.generated_tokens:
            return self.generated_tokens[-1]
        return self.prompt_tokens[-1]


    @property
    def is_prefill_complete(self) -> bool:
        """Returns: True if all prompt tokens have been prefilled"""
        return self.num_prefilled_tokens >= self.prompt_length


    @property
    def pending_tokens(self) -> list[int]:
        """Returns: List of pending token IDs (not yet prefilled)"""
        return self.prompt_tokens[self.num_prefilled_tokens:]


    def append_token(self, token_id: int) -> None:
        """Param token_id: Token ID to append to the sequence"""
        self.generated_tokens.append(token_id)


    def advance_prefill(self, num_tokens: int) -> None:
        """Param num_tokens: Number of tokens prefilled"""
        self.num_prefilled_tokens += num_tokens
        if self.is_prefill_complete:
            self.status = SequenceStatus.RUNNING


    def reset_for_preemption(self) -> None:
        """Reset sequence state for recomputation after preemption"""
        self.num_prefilled_tokens = 0
        self.num_cached_tokens = 0
        self.block_table.clear()
        self.conv_state_slot = -1
        self.status = SequenceStatus.WAITING


    def get_num_required_blocks(self, block_size: int) -> int:
        """Calculate blocks needed for current sequence length.

        :param block_size: Tokens per block
        :returns: Number of blocks required
        """
        return (self.current_length + block_size - 1) // block_size


    def __repr__(self) -> str:
        """Return string representation of the sequence"""
        return (
            f"Sequence(id={self.seq_id}, "
            f"prompt={self.prompt_length}, "
            f"generated={self.generated_length}, "
            f"status={self.status.name})"
        )
