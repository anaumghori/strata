import torch
from torch import Tensor

from strata.config import StrataConfig
from strata.engine.sequence import Sequence
from strata.engine.sampling import Sampler
from strata.cache.state_manager import StateManager
from strata.models.base import BaseModelForCausalLM


class ModelRunner:
    """
    Executes model forward passes and handles batch preparation. Coordinates between the 
    scheduler, model, and state manager to run prefill and decode steps efficiently on CPU.
    """

    def __init__(
        self,
        model: BaseModelForCausalLM,
        state_manager: StateManager,
        config: StrataConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize model runner with model and state manager.

        :param model: The causal language model
        :param state_manager: Cache state manager
        :param config: Engine configuration
        :param dtype: Data type for computation
        """
        self.model = model
        self.state_manager = state_manager
        self.config = config
        self.dtype = dtype
        self.block_size = config.block_size
        self.prefill_chunk_size = config.prefill_chunk_size

        self.sampler = Sampler()


    def run_prefill(self, sequences: list[Sequence]) -> list[int]:
        """Execute prefill step for a batch of sequences.

        :param sequences: Sequences to prefill
        :returns: Sampled token IDs (one per sequence)
        """
        (input_ids, positions, seq_lens, context_lens, slot_mapping, 
         block_tables, conv_slots) = self._prepare_prefill(sequences)
        kv_caches = self.state_manager.get_kv_caches()
        conv_states = self.state_manager.get_conv_states()
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.long)

        with torch.inference_mode():
            logits, new_conv_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                slot_mapping=slot_mapping,
                context_lens=context_lens_tensor,
                block_tables=block_tables,
                seq_lens=seq_lens,
                is_prefill=True,
                conv_states=conv_states,
                seq_ids=conv_slots,
            )

        if new_conv_states:
            self._update_conv_states(sequences, new_conv_states, seq_lens)
        last_token_logits = self._extract_last_token_logits(logits, seq_lens)
        temperatures = torch.tensor(
            [seq.temperature for seq in sequences],
            dtype=torch.float32,
        )

        token_ids = self.sampler.sample(
            last_token_logits,
            temperatures,
            top_k=sequences[0].top_k if sequences else None,
            top_p=sequences[0].top_p if sequences else None,
        )

        return token_ids.tolist()


    def run_decode(self, sequences: list[Sequence]) -> list[int]:
        """Execute decode step for a batch of sequences.

        :param sequences: Sequences in decode phase
        :returns: Sampled token IDs (one per sequence)
        """
        input_ids, positions, context_lens, slot_mapping, block_tables, conv_slots = (self._prepare_decode(sequences))
        kv_caches = self.state_manager.get_kv_caches()
        conv_states = self.state_manager.get_conv_states()
        with torch.inference_mode():
            logits, new_conv_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                is_prefill=False,
                conv_states=conv_states,
                seq_ids=conv_slots,
            )

        if new_conv_states:
            self._update_decode_conv_states(sequences, new_conv_states)

        temperatures = torch.tensor(
            [seq.temperature for seq in sequences],
            dtype=torch.float32,
        )

        token_ids = self.sampler.sample(
            logits,
            temperatures,
            top_k=sequences[0].top_k if sequences else None,
            top_p=sequences[0].top_p if sequences else None,
        )

        return token_ids.tolist()


    def _prepare_prefill(self, sequences: list[Sequence]) -> tuple[Tensor, Tensor, list[int], list[int], Tensor, Tensor, Tensor]:
        """Prepare input tensors for prefill step.

        :param sequences: Sequences to prefill
        :returns: Tuple of (input_ids, positions, seq_lens, context_lens, slot_mapping, block_tables, conv_slots)
        """
        all_input_ids = []
        all_positions = []
        seq_lens = []
        context_lens = []
        all_slots = []
        conv_slots = []

        max_blocks = max((len(seq.block_table) for seq in sequences), default=0)
        block_tables = []

        for seq in sequences:
            pending = seq.pending_tokens
            chunk_size = min(len(pending), self.prefill_chunk_size)
            tokens = pending[:chunk_size]
            start_pos = seq.num_prefilled_tokens
            end_pos = start_pos + chunk_size
            all_input_ids.extend(tokens)
            all_positions.extend(range(start_pos, end_pos))
            seq_lens.append(chunk_size)
            context_lens.append(end_pos)
            slots = self.state_manager.get_slot_mapping(seq.seq_id, start_pos, end_pos)
            all_slots.extend(slots)
            conv_slots.append(self.state_manager.get_conv_slot(seq.seq_id))
            bt = seq.block_table + [-1] * (max_blocks - len(seq.block_table))
            block_tables.append(bt)

        return (
            torch.tensor(all_input_ids, dtype=torch.long),
            torch.tensor(all_positions, dtype=torch.long),
            seq_lens,
            context_lens,
            torch.tensor(all_slots, dtype=torch.long),
            torch.tensor(block_tables, dtype=torch.long),
            torch.tensor(conv_slots, dtype=torch.long),
        )


    def _prepare_decode(self, sequences: list[Sequence]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Prepare input tensors for decode step.

        :param sequences: Sequences in decode phase
        :returns: Tuple of (input_ids, positions, context_lens, slot_mapping, block_tables, conv_slots)
        """
        input_ids = []
        positions = []
        context_lens = []
        slot_mapping = []
        conv_slots = []
        max_blocks = max((len(seq.block_table) for seq in sequences), default=0)
        block_tables = []

        for seq in sequences:
            input_ids.append(seq.last_token)
            positions.append(seq.current_length - 1)
            context_lens.append(seq.current_length)

            slot = self.state_manager.get_slot_mapping(
                seq.seq_id,
                seq.current_length - 1,
                seq.current_length,
            )
            slot_mapping.extend(slot)
            conv_slots.append(self.state_manager.get_conv_slot(seq.seq_id))
            bt = seq.block_table + [-1] * (max_blocks - len(seq.block_table))
            block_tables.append(bt)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
            torch.tensor(context_lens, dtype=torch.long),
            torch.tensor(slot_mapping, dtype=torch.long),
            torch.tensor(block_tables, dtype=torch.long),
            torch.tensor(conv_slots, dtype=torch.long),
        )


    def _extract_last_token_logits(
        self,
        logits: Tensor,
        seq_lens: list[int],
    ) -> Tensor:
        """
        :param logits: Full logits tensor of shape [total_tokens, vocab_size]
        :param seq_lens: Length of each sequence in the batch
        :returns: Logits of shape [num_sequences, vocab_size]
        """
        last_logits = []
        offset = 0
        for seq_len in seq_lens:
            last_idx = offset + seq_len - 1
            last_logits.append(logits[last_idx])
            offset += seq_len

        return torch.stack(last_logits, dim=0)


    def _update_conv_states(
        self,
        sequences: list[Sequence],
        new_conv_states: list[Tensor],
        seq_lens: list[int],
    ) -> None:
        """
        :param sequences: Sequences that were prefilled
        :param new_conv_states: New state values per ShortConv layer
        :param seq_lens: Chunk sizes for each sequence
        """
        seq_ids = [seq.seq_id for seq in sequences]
        for layer_idx, states in enumerate(new_conv_states):
            self.state_manager.update_conv_state(layer_idx, seq_ids, states)


    def _update_decode_conv_states(
        self,
        sequences: list[Sequence],
        new_conv_states: list[Tensor],
    ) -> None:
        """
        :param sequences: Sequences that were decoded
        :param new_conv_states: New state values per ShortConv layer
        """
        seq_ids = [seq.seq_id for seq in sequences]
        slots = torch.tensor(
            [self.state_manager.get_conv_slot(sid) for sid in seq_ids],
            dtype=torch.long,
        )

        for layer_idx, states in enumerate(new_conv_states):
            self.state_manager.conv_state.set_state(layer_idx, slots, states)
