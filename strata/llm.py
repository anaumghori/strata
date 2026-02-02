from typing import Iterator, Optional

from strata.config import StrataConfig, SamplingParams
from strata.engine.engine import LLMEngine
from strata.utils.chat_template import ChatMessage


class LLM:
    """High-level API for LLM inference on CPU"""

    def __init__(
        self,
        model_path: str,
        dtype: str = "auto",
        max_num_seqs: int = 256,
        max_context_length: int = 16384,
        block_size: int = 16,
        max_num_batched_tokens: int = 4096,
        prefill_chunk_size: int = 256,
        num_threads: Optional[int] = None,
        use_compile: bool = False,
        **kwargs,
    ) -> None:
        """Initialize LLM with model and configuration.

        :param model_path: Path to HuggingFace model directory
        :param dtype: Data type ('auto', 'bf16', 'fp32')
        :param max_num_seqs: Maximum concurrent sequences
        :param max_context_length: Maximum context length per sequence
        :param block_size: KV cache block size
        :param max_num_batched_tokens: Maximum tokens per batch
        :param prefill_chunk_size: Tokens per prefill chunk
        :param num_threads: Number of CPU threads (auto if None)
        :param use_compile: Enable torch.compile for decode
        :param kwargs: Additional configuration options
        """
        config = StrataConfig(
            model_path=model_path,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
            max_context_length=max_context_length,
            block_size=block_size,
            max_num_batched_tokens=max_num_batched_tokens,
            prefill_chunk_size=prefill_chunk_size,
            num_threads=num_threads,
            use_compile=use_compile,
            memory_utilization=kwargs.get("memory_utilization", 0.9),
            prefill_fairness_chunks=kwargs.get("prefill_fairness_chunks", 4),
            enable_prefix_caching=kwargs.get("enable_prefix_caching", True),
            default_temperature=kwargs.get("default_temperature", 1.0),
            default_max_tokens=kwargs.get("default_max_tokens", 256),
        )

        self.engine = LLMEngine(config)
        self.config = config


    def generate(
        self,
        prompts: str | list[str],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ) -> list[dict]:
        """Generate completions for one or more prompts.

        :param prompts: Single prompt string or list of prompts
        :param sampling_params: Generation parameters
        :param kwargs: Overrides for sampling params
        :returns: List of output dictionaries with text and token IDs
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", self.config.default_temperature),
                max_tokens=kwargs.get("max_tokens", self.config.default_max_tokens),
                top_k=kwargs.get("top_k", None),
                top_p=kwargs.get("top_p", None),
                stop_sequences=kwargs.get("stop_sequences", None),
            )

        return self.engine.generate(prompts, sampling_params)


    def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Generate tokens with streaming output.

        :param prompt: Input prompt string
        :param sampling_params: Generation parameters
        :param kwargs: Overrides for sampling params
        :returns: Iterator yielding generated token strings
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", self.config.default_temperature),
                max_tokens=kwargs.get("max_tokens", self.config.default_max_tokens),
                top_k=kwargs.get("top_k", None),
                top_p=kwargs.get("top_p", None),
                stop_sequences=kwargs.get("stop_sequences", None),
            )

        yield from self.engine.generate_stream(prompt, sampling_params)


    def __call__(
        self,
        prompts: str | list[str],
        **kwargs,
    ) -> list[dict]:
        """
        :param prompts: Input prompts
        :param kwargs: Generation parameters
        :returns: List of output dictionaries
        """
        return self.generate(prompts, **kwargs)


    def chat(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        sampling_params: Optional[SamplingParams] = None,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Automatically applies the appropriate chat template to convert
        messages into model-ready prompts. Uses the tokenizer's built-in
        template if available, otherwise falls back to ChatML format.

        :param messages: Single conversation or list of conversations
        :param sampling_params: Generation parameters
        :param add_generation_prompt: Whether to add assistant prompt
        :param chat_template: Optional explicit template override
        :param kwargs: Overrides for sampling params
        :returns: List of output dictionaries with text and token IDs
        """
        if not messages:
            return []

        if isinstance(messages[0], dict):
            conversations = [messages]
        else:
            conversations = messages

        prompts = []
        for conversation in conversations:
            prompt = self.engine.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                chat_template=chat_template,
            )
            prompts.append(prompt)

        return self.generate(prompts, sampling_params, **kwargs)


    def apply_chat_template(
        self,
        messages: list[ChatMessage],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
    ) -> str | list[int]:
        """
        :param messages: List of chat messages with role and content
        :param tokenize: If True, return token IDs instead of string
        :param add_generation_prompt: Whether to add assistant prompt
        :param chat_template: Optional explicit template override
        :returns: Formatted prompt string or token IDs
        """
        return self.engine.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
        )
