import torch
from torch import Tensor
import torch.nn.functional as F


class Sampler:
    """
    Token sampling implementation for text generation. Supports greedy decoding, temperature scaling, and
    optional top-k and top-p filtering.
    """

    def __init__(self) -> None:
        pass


    def sample(
        self,
        logits: Tensor,
        temperatures: Tensor,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Tensor:
        """Sample tokens from logits using specified parameters.

        :param logits: Logits tensor of shape [batch_size, vocab_size]
        :param temperatures: Temperature per sequence of shape [batch_size]
        :param top_k: Optional top-k filtering threshold
        :param top_p: Optional nucleus sampling threshold
        :returns: Sampled token IDs of shape [batch_size]
        """
        logits = logits.float()
        greedy_mask = temperatures == 0
        safe_temps = temperatures.clamp(min=1e-7)
        logits = logits / safe_temps.unsqueeze(-1)

        if top_k is not None and top_k > 0:
            logits = self._apply_top_k(logits, top_k)

        if top_p is not None and top_p < 1.0:
            logits = self._apply_top_p(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        if greedy_mask.any():
            greedy_tokens = logits.argmax(dim=-1)
            sampled = torch.where(greedy_mask, greedy_tokens, sampled)
        return sampled


    def _apply_top_k(self, logits: Tensor, top_k: int) -> Tensor:
        """Apply top-k filtering to logits.

        :param logits: Logits tensor of shape [batch_size, vocab_size]
        :param top_k: Number of top tokens to keep
        :returns: Filtered logits tensor
        """
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, float("-inf"), logits)
        return logits


    def _apply_top_p(self, logits: Tensor, top_p: float) -> Tensor:
        """Apply nucleus (top-p) filtering to logits.

        :param logits: Logits tensor of shape [batch_size, vocab_size]
        :param top_p: Cumulative probability threshold
        :returns: Filtered logits tensor
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        sorted_logits[mask] = float("-inf")
        original_logits = torch.gather(
            sorted_logits,
            dim=-1,
            index=sorted_indices.argsort(dim=-1),
        )
        return original_logits
