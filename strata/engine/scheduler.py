from collections import deque
from dataclasses import dataclass
from typing import Optional

from strata.config import StrataConfig
from strata.engine.sequence import Sequence, SequenceStatus
from strata.cache.state_manager import StateManager


@dataclass
class SchedulerOutput:
    """
    Output from a scheduler step. Contains the scheduled sequences and metadata about
    whether this is a prefill or decode step.
    """
    sequences: list[Sequence]
    is_prefill: bool
    num_tokens: int


class Scheduler:
    """
    Continuous batching scheduler with prefill-decode separation. Manages waiting and running queues, 
    implements chunked prefill, and handles admission control and preemption.
    """

    def __init__(
        self,
        config: StrataConfig,
        state_manager: StateManager,
        eos_token_id: int,
    ) -> None:
        """
        :param config: Engine configuration
        :param state_manager: Cache state manager
        :param eos_token_id: End-of-sequence token ID
        """
        self.config = config
        self.state_manager = state_manager
        self.eos_token_id = eos_token_id
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.prefill_chunk_size = config.prefill_chunk_size
        self.prefill_fairness_chunks = config.prefill_fairness_chunks
        self.block_size = config.block_size
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.consecutive_prefill_chunks = 0


    def add(self, seq: Sequence) -> None:
        """
        :param seq: Sequence to add to the waiting queue.
        :returns: None
        """
        self.waiting.append(seq)


    def is_finished(self) -> bool:
        """Returns: True if both queues are empty (if all sequences have completed)"""
        return not self.waiting and not self.running


    def schedule(self) -> Optional[SchedulerOutput]:
        """Returns: SchedulerOutput with scheduled sequences or None"""
        should_prefill = self._should_schedule_prefill()
        if should_prefill:
            return self._schedule_prefill()
        else:
            return self._schedule_decode()


    def _should_schedule_prefill(self) -> bool:
        """Returns: True if prefill should be scheduled"""
        if not self.waiting:
            return False

        if not self.running:
            return True

        if self.consecutive_prefill_chunks >= self.prefill_fairness_chunks:
            return False

        return True


    def _schedule_prefill(self) -> Optional[SchedulerOutput]:
        """Returns: SchedulerOutput for prefill or None"""
        scheduled_seqs = []
        num_tokens = 0

        while self.waiting:
            seq = self.waiting[0]

            if len(scheduled_seqs) >= self.max_num_seqs:
                break

            pending = seq.pending_tokens
            tokens_to_add = min(len(pending), self.prefill_chunk_size)

            if num_tokens + tokens_to_add > self.max_num_batched_tokens:
                break

            if not seq.block_table:
                if not self.state_manager.can_allocate(seq.prompt_length + 1):
                    break
                if not self._allocate_sequence(seq):
                    break

            scheduled_seqs.append(seq)
            num_tokens += tokens_to_add
            self.waiting.popleft()

        if not scheduled_seqs:
            self.consecutive_prefill_chunks = 0
            return self._schedule_decode()

        self.consecutive_prefill_chunks += 1

        return SchedulerOutput(
            sequences=scheduled_seqs,
            is_prefill=True,
            num_tokens=num_tokens,
        )


    def _schedule_decode(self) -> Optional[SchedulerOutput]:
        """Returns: SchedulerOutput for decode or None"""
        self.consecutive_prefill_chunks = 0
        if not self.running:
            return None

        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            can_append = self.state_manager.can_append_token(seq.seq_id, seq.current_length)

            if not can_append:
                if self.running:
                    victim = self.running.pop()
                    self._preempt(victim)
                    self.running.appendleft(seq)
                    continue
                else:
                    self._preempt(seq)
                    break

            self.state_manager.allocate_for_append(seq.seq_id, seq.current_length)
            seq.block_table = self.state_manager.get_block_table(seq.seq_id)
            scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        if not scheduled_seqs:
            return None

        return SchedulerOutput(
            sequences=scheduled_seqs,
            is_prefill=False,
            num_tokens=len(scheduled_seqs),
        )


    def _allocate_sequence(self, seq: Sequence) -> bool:
        """
        :param seq: Sequence to allocate cache resources for.
        :returns: True if allocation succeeded
        """
        success = self.state_manager.allocate_sequence(
            seq.seq_id,
            seq.prompt_length,
            prompt_tokens=seq.prompt_tokens,
        )
        if success:
            seq.block_table = self.state_manager.get_block_table(seq.seq_id)
            seq.conv_state_slot = self.state_manager.get_conv_slot(seq.seq_id)
            seq.num_cached_tokens = self.state_manager.get_num_cached_tokens(seq.seq_id)
            seq.num_prefilled_tokens = seq.num_cached_tokens
        return success


    def _preempt(self, seq: Sequence) -> None:
        """Param seq: Sequence to preempt by deallocating and requeueing"""
        self.state_manager.deallocate_sequence(seq.seq_id)
        seq.reset_for_preemption()
        self.waiting.appendleft(seq)


    def postprocess(
        self,
        scheduled_seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ) -> list[Sequence]:
        """Process generated tokens and update sequence states.

        :param scheduled_seqs: Sequences that were executed
        :param token_ids: Generated token IDs (one per sequence)
        :param is_prefill: Whether this was a prefill step
        :returns: List of finished sequences
        """
        finished = []

        if is_prefill:
            for seq, token_id in zip(scheduled_seqs, token_ids):
                pending_count = len(seq.pending_tokens)
                chunk_size = min(pending_count, self.prefill_chunk_size)
                seq.advance_prefill(chunk_size)

                if seq.is_prefill_complete:
                    seq.append_token(token_id)
                    self.state_manager.allocate_for_append(
                        seq.seq_id, seq.current_length
                    )
                    seq.block_table = self.state_manager.get_block_table(seq.seq_id)
                    should_finish = self._check_finish_condition(seq, token_id)
                    if should_finish:
                        seq.status = SequenceStatus.FINISHED
                        self.state_manager.deallocate_sequence(seq.seq_id)
                        finished.append(seq)
                    else:
                        self.running.append(seq)
                else:
                    self.waiting.appendleft(seq)
        else:
            for seq, token_id in zip(scheduled_seqs, token_ids):
                seq.append_token(token_id)

                should_finish = self._check_finish_condition(seq, token_id)
                if should_finish:
                    seq.status = SequenceStatus.FINISHED
                    self.state_manager.deallocate_sequence(seq.seq_id)
                    self.running.remove(seq)
                    finished.append(seq)

        return finished


    def _check_finish_condition(self, seq: Sequence, token_id: int) -> bool:
        """
        :param seq: Sequence to check
        :param token_id: Most recently generated token
        :returns: True if sequence should finish generation.
        """
        eos_hit = not seq.ignore_eos and token_id == self.eos_token_id
        max_reached = seq.generated_length >= seq.max_new_tokens
        return eos_hit or max_reached
