import torch
from torch import nn, Tensor


class RotaryEmbedding(nn.Module):
    """
    Computes and applies rotary embeddings to query and key tensors
    using the rotate_half pattern 
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 1000000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        :param head_dim: Dimension per attention head
        :param max_position_embeddings: Maximum sequence length
        :param base: Base frequency for RoPE computation
        :param dtype: Data type for the embedding tensors
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_cached = emb.cos().to(dtype)
        sin_cached = emb.sin().to(dtype)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)


    def forward(
        self,
        positions: Tensor,
        q: Tensor,
        k: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        :param positions: Position indices of shape [num_tokens]
        :param q: Query tensor of shape [num_tokens, num_heads, head_dim]
        :param k: Key tensor of shape [num_tokens, num_kv_heads, head_dim]
        :returns: Tuple of rotated (query, key) tensors
        """
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        if cos.dim() == 2:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = self._apply_rotary(q, cos, sin)
        k_embed = self._apply_rotary(k, cos, sin)
        return q_embed, k_embed


    def _apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        :param x: Input tensor to rotate
        :param cos: Cosine values for rotation
        :param sin: Sine values for rotation
        :returns: Rotated tensor
        """
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)
        return x_rotated


    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        :param x: Input tensor
        :returns: Tensor with rotated dimensions
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
