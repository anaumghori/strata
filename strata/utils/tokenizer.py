from transformers import AutoTokenizer

from strata.utils.chat_template import ChatMessage, apply_chat_template


class Tokenizer:
    """
    Wrapper for HuggingFace tokenizer. Provides encoding, decoding, and chat 
    template methods optimized for inference pipeline integration.
    """

    def __init__(self, model_path: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )


    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id


    @property
    def bos_token_id(self) -> int | None:
        return self._tokenizer.bos_token_id


    @property
    def pad_token_id(self) -> int | None:
        return self._tokenizer.pad_token_id


    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer) # Number of tokens in vocabulary


    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        :param text: Input text string
        :param add_special_tokens: Whether to add BOS/EOS tokens
        :returns: List of token IDs
        """
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )


    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """D
        :param token_ids: List of token IDs
        :param skip_special_tokens: Whether to skip special tokens
        :returns: Decoded text string
        """
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )


    def batch_encode(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        return [
            self.encode(text, add_special_tokens=add_special_tokens)
            for text in texts
        ]


    def batch_decode(self, token_ids_list: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in token_ids_list
        ]


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
        return apply_chat_template(
            self._tokenizer,
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
        )


    @property
    def has_chat_template(self) -> bool:
        """Returns: True if tokenizer has a built-in chat template"""
        return hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template is not None
