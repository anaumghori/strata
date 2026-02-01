from typing import Iterator, Optional
import torch

from strata.config import StrataConfig, SamplingParams, get_dtype
from strata.engine.sequence import Sequence
from strata.engine.scheduler import Scheduler
from strata.engine.model_runner import ModelRunner
from strata.cache.state_manager import StateManager
from strata.models.registry import build_model
from strata.utils.tokenizer import Tokenizer


class LLMEngine:
    """
    Main inference engine for continuous batching execution. Coordinates model loading, 
    tokenization, scheduling, and generation for concurrent sequence processing.
    """

    def __init__(self, config: StrataConfig) -> None:
        self.config = config
        self.dtype = get_dtype(config.dtype)
        self._setup_threads()
        self.tokenizer = Tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.model = build_model(config.model_path, self.dtype)
        self.state_manager = self._create_state_manager()

        self.scheduler = Scheduler(
            config=config,
            state_manager=self.state_manager,
            eos_token_id=self.eos_token_id,
        )

        self.model_runner = ModelRunner(
            model=self.model,
            state_manager=self.state_manager,
            config=config,
            dtype=self.dtype,
        )


    def _setup_threads(self) -> None:
        """Configure PyTorch thread settings for CPU inference"""
        if self.config.num_threads is not None:
            torch.set_num_threads(self.config.num_threads)
        torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision("high")


    def _create_state_manager(self) -> StateManager:
        num_blocks = self._compute_num_blocks()
        return StateManager(
            num_attention_layers=self.model.num_attention_layers,
            num_shortconv_layers=self.model.num_shortconv_layers,
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            num_kv_heads=self.model.num_kv_heads,
            head_dim=self.model.head_dim,
            max_sequences=self.config.max_num_seqs,
            conv_kernel_size=self.model.conv_kernel_size,
            conv_dim=self.model.conv_dim,
            dtype=self.dtype,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )


    def _compute_num_blocks(self) -> int:
        """Returns: Number of KV cache blocks to allocate"""
        max_tokens = self.config.max_num_seqs * self.config.max_context_length
        max_blocks = max_tokens // self.config.block_size
        target_blocks = int(max_blocks * self.config.memory_utilization)
        min_blocks = self.config.max_num_seqs * 4
        return min(max(target_blocks, min_blocks), max_blocks)


    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> int:
        """
        :param prompt: Input prompt as string or token IDs
        :param sampling_params: Generation parameters
        :returns: Sequence ID for tracking
        """
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.config.default_temperature,
                max_tokens=self.config.default_max_tokens,
            )

        seq = Sequence(token_ids, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id


    def step(self) -> tuple[list[dict], int]:
        """Execute one scheduling and inference step.

        :returns: Tuple of (finished outputs, token count processed)
        """
        schedule_output = self.scheduler.schedule()
        if schedule_output is None:
            return [], 0

        sequences = schedule_output.sequences
        if schedule_output.is_prefill:
            token_ids = self.model_runner.run_prefill(sequences)
        else:
            token_ids = self.model_runner.run_decode(sequences)

        finished_seqs = self.scheduler.postprocess(
            sequences, token_ids, schedule_output.is_prefill
        )
        outputs = []
        for seq in finished_seqs:
            text = self.tokenizer.decode(seq.generated_tokens)
            outputs.append({
                "seq_id": seq.seq_id,
                "text": text,
                "token_ids": seq.generated_tokens,
                "prompt_tokens": seq.prompt_tokens,
            })
        return outputs, schedule_output.num_tokens


    def is_finished(self) -> bool:
        """Returns: True if no pending or running requests"""
        return self.scheduler.is_finished()


    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: Optional[SamplingParams | list[SamplingParams]] = None,
    ) -> list[dict]:
        """Generate completions for a batch of prompts.

        :param prompts: List of prompts as strings or token IDs
        :param sampling_params: Generation parameters (single or per-prompt)
        :returns: List of output dictionaries with text and token IDs
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.config.default_temperature,
                max_tokens=self.config.default_max_tokens,
            )

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        seq_ids = []
        for prompt, sp in zip(prompts, sampling_params):
            seq_id = self.add_request(prompt, sp)
            seq_ids.append(seq_id)
        outputs_dict = {}
        while not self.is_finished():
            outputs, _ = self.step()
            for output in outputs:
                outputs_dict[output["seq_id"]] = output

        results = [outputs_dict[seq_id] for seq_id in seq_ids]
        return results


    def generate_stream(
        self,
        prompt: str | list[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> Iterator[str]:
        """Generate tokens with streaming output.

        :param prompt: Input prompt as string or token IDs
        :param sampling_params: Generation parameters
        :returns: Iterator yielding generated token strings
        """
        self.add_request(prompt, sampling_params)

        last_length = 0

        while not self.is_finished():
            outputs, _ = self.step()
            for output in outputs:
                tokens = output["token_ids"]
                new_tokens = tokens[last_length:]
                last_length = len(tokens)
                for token_id in new_tokens:
                    yield self.tokenizer.decode([token_id])
