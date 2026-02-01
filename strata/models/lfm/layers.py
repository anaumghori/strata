from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from strata.models.lfm.config import LfmConfig
from strata.models.lfm.rope import RotaryEmbedding


class RMSNorm(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        :param hidden_size: Size of the hidden dimension
        :param eps: Small constant for numerical stability
        :param dtype: Data type for the weight parameter
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))


    def forward(
        self,
        x: Tensor,
        residual: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply RMSNorm with optional fused residual addition.

        :param x: Input tensor of shape [..., hidden_size]
        :param residual: Optional residual tensor for fused add
        :returns: Normalized tensor, or tuple of (normalized, new_residual)
        """
        if residual is not None:
            x = x + residual
            residual = x

        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(orig_dtype) * self.weight
        if residual is not None:
            return x, residual
        return x


class Lfm2Attention(nn.Module):
    """Grouped Query Attention with QK normalization for LFM2"""

    def __init__(
        self,
        config: LfmConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize attention layer with projections and norms.

        :param config: LFM model configuration
        :param layer_idx: Index of this layer in the model
        :param dtype: Data type for weight tensors
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            dtype=dtype,
        )

        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps, dtype=dtype)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            dtype=dtype,
        )


    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        kv_cache: Optional[Tensor] = None,
        slot_mapping: Optional[Tensor] = None,
        context_lens: Optional[Tensor] = None,
        block_table: Optional[Tensor] = None,
        seq_lens: Optional[list[int]] = None,
        is_prefill: bool = True,
        attn_layer_idx: int = 0,
    ) -> Tensor:
        """Execute attention forward pass with optional KV caching.

        :param hidden_states: Input tensor of shape [num_tokens, hidden_size]
        :param positions: Position indices of shape [num_tokens]
        :param kv_cache: Optional KV cache tensor for this layer
        :param slot_mapping: Mapping of tokens to cache slots
        :param context_lens: Context lengths per sequence (total context including cached)
        :param block_table: Block table for paged attention
        :param seq_lens: Sequence lengths for prefill (current chunk sizes)
        :param is_prefill: Whether this is prefill or decode
        :param attn_layer_idx: Index within attention layers (0-5)
        :returns: Output tensor of shape [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)
        q, k = self.rotary_emb(positions, q, k)
        if kv_cache is not None and slot_mapping is not None:
            k_cache, v_cache = kv_cache
            self._write_to_cache(k, v, k_cache, v_cache, slot_mapping)

        if is_prefill:
            output = self._prefill_attention(q, k, v, seq_lens, context_lens, block_table, kv_cache)
        else:
            output = self._decode_attention(q, kv_cache, context_lens, block_table)
        output = output.view(num_tokens, self.num_heads * self.head_dim)
        output = self.out_proj(output)

        return output


    def _write_to_cache(
        self,
        k: Tensor,
        v: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
    ) -> None:
        """Write key-value pairs to the paged cache.

        :param k: Key tensor of shape [num_tokens, num_kv_heads, head_dim]
        :param v: Value tensor of shape [num_tokens, num_kv_heads, head_dim]
        :param k_cache: Key cache of shape [num_blocks, block_size, num_kv_heads, head_dim]
        :param v_cache: Value cache of shape [num_blocks, block_size, num_kv_heads, head_dim]
        :param slot_mapping: Mapping of tokens to cache slots
        :returns: None
        """
        num_blocks = k_cache.shape[0]
        block_size = k_cache.shape[1]

        for i, slot in enumerate(slot_mapping):
            if slot >= 0:
                block_idx = slot // block_size
                offset = slot % block_size
                k_cache[block_idx, offset] = k[i]
                v_cache[block_idx, offset] = v[i]


    def _prefill_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        seq_lens: Optional[list[int]],
        context_lens: Optional[Tensor] = None,
        block_table: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Compute attention for prefill using SDPA with padding.

        :param q: Query tensor of shape [num_tokens, num_heads, head_dim]
        :param k: Key tensor of shape [num_tokens, num_kv_heads, head_dim]
        :param v: Value tensor of shape [num_tokens, num_kv_heads, head_dim]
        :param seq_lens: List of chunk sizes in the batch (current input lengths)
        :param context_lens: Total context lengths per sequence (including cached tokens)
        :param block_table: Block table for reading cached K,V
        :param kv_cache: KV cache tuple for reading cached values
        :returns: Attention output of shape [num_tokens, num_heads, head_dim]
        """
        if seq_lens is None:
            seq_lens = [q.shape[0]]

        k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
        v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)

        outputs = []
        offset = 0

        for seq_idx, seq_len in enumerate(seq_lens):
            q_seq = q[offset : offset + seq_len]
            k_seq = k_expanded[offset : offset + seq_len]
            v_seq = v_expanded[offset : offset + seq_len]
            ctx_len = context_lens[seq_idx].item() if context_lens is not None else seq_len
            cached_len = ctx_len - seq_len

            if cached_len > 0 and kv_cache is not None and block_table is not None:
                k_cache, v_cache = kv_cache
                block_size = k_cache.shape[1]
                blocks = block_table[seq_idx]
                k_cached = self._gather_from_cache(k_cache, blocks, cached_len, block_size)
                v_cached = self._gather_from_cache(v_cache, blocks, cached_len, block_size)
                k_cached = k_cached.repeat_interleave(self.num_kv_groups, dim=1)
                v_cached = v_cached.repeat_interleave(self.num_kv_groups, dim=1)
                k_seq = torch.cat([k_cached, k_seq], dim=0)
                v_seq = torch.cat([v_cached, v_seq], dim=0)

            q_seq = q_seq.transpose(0, 1).unsqueeze(0)
            k_seq = k_seq.transpose(0, 1).unsqueeze(0)
            v_seq = v_seq.transpose(0, 1).unsqueeze(0)

            if cached_len > 0:
                q_len = seq_len
                kv_len = ctx_len
                q_indices = torch.arange(q_len, device=q.device).unsqueeze(1)
                kv_indices = torch.arange(kv_len, device=q.device).unsqueeze(0)
                attn_mask = kv_indices <= (cached_len + q_indices)
                
                out = F.scaled_dot_product_attention(
                    q_seq, k_seq, v_seq,
                    attn_mask=attn_mask,
                    scale=self.scaling,
                )
            else:
                out = F.scaled_dot_product_attention(
                    q_seq, k_seq, v_seq,
                    is_causal=True,
                    scale=self.scaling,
                )

            out = out.squeeze(0).transpose(0, 1)
            outputs.append(out)
            offset += seq_len

        return torch.cat(outputs, dim=0)


    def _decode_attention(
        self,
        q: Tensor,
        kv_cache: tuple[Tensor, Tensor],
        context_lens: Tensor,
        block_table: Tensor,
    ) -> Tensor:
        """Compute attention for decode by gathering from KV cache.

        :param q: Query tensor of shape [batch_size, num_heads, head_dim]
        :param kv_cache: Tuple of (k_cache, v_cache) tensors for this layer
        :param context_lens: Context length per sequence
        :param block_table: Block table mapping logical to physical blocks
        :returns: Attention output of shape [batch_size, num_heads, head_dim]
        """
        batch_size = q.shape[0]
        k_cache, v_cache = kv_cache
        block_size = k_cache.shape[1]
        outputs = []

        for i in range(batch_size):
            ctx_len = context_lens[i].item()
            blocks = block_table[i]
            k_seq = self._gather_from_cache(k_cache, blocks, ctx_len, block_size)
            v_seq = self._gather_from_cache(v_cache, blocks, ctx_len, block_size)
            k_seq = k_seq.repeat_interleave(self.num_kv_groups, dim=1)
            v_seq = v_seq.repeat_interleave(self.num_kv_groups, dim=1)
            q_i = q[i : i + 1]
            q_i = q_i.transpose(0, 1).unsqueeze(0)
            k_seq = k_seq.transpose(0, 1).unsqueeze(0)
            v_seq = v_seq.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q_i, k_seq, v_seq,
                is_causal=False,
                scale=self.scaling,
            )

            out = out.squeeze(0).transpose(0, 1)
            outputs.append(out)

        return torch.cat(outputs, dim=0)


    def _gather_from_cache(
        self,
        cache: Tensor,
        block_ids: Tensor,
        length: int,
        block_size: int,
    ) -> Tensor:
        """Gather KV values from paged cache for a single sequence.

        :param cache: Cache tensor of shape [num_blocks, block_size, num_kv_heads, head_dim]
        :param block_ids: Block IDs for this sequence
        :param length: Number of tokens to gather
        :param block_size: Tokens per block
        :returns: Gathered tensor of shape [length, num_kv_heads, head_dim]
        """
        gathered = []
        remaining = length
        for block_idx in block_ids:
            if remaining <= 0:
                break
            block_id = block_idx.item()
            if block_id < 0:
                break
            tokens_in_block = min(remaining, block_size)
            gathered.append(cache[block_id, :tokens_in_block])
            remaining -= tokens_in_block

        return torch.cat(gathered, dim=0)


class Lfm2ShortConv(nn.Module):
    """
    Gated Short Convolution layer for LFM2.

    Implements depthwise causal conv1d with BCx gating pattern,
    supporting both prefill and incremental decode modes. Dimensions
    are derived dynamically from the model configuration.
    """

    def __init__(
        self,
        config: LfmConfig,
        layer_idx: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize ShortConv layer with projections and conv weight.

        :param config: LFM model configuration
        :param layer_idx: Index of this layer in the model
        :param dtype: Data type for weight tensors
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_dim = config.conv_dim
        self.kernel_size = config.conv_kernel_size

        self.in_proj = nn.Linear(
            config.hidden_size,
            3 * self.conv_dim,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            self.conv_dim,
            config.hidden_size,
            bias=False,
            dtype=dtype,
        )

        self.conv_weight = nn.Parameter(
            torch.empty(self.conv_dim, 1, self.kernel_size, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.conv_weight)


    def forward(
        self,
        hidden_states: Tensor,
        conv_state: Optional[Tensor] = None,
        seq_ids: Optional[Tensor] = None,
        seq_lens: Optional[list[int]] = None,
        is_prefill: bool = True,
        shortconv_layer_idx: int = 0,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Execute ShortConv forward pass with optional state management.

        :param hidden_states: Input tensor of shape [num_tokens, hidden_size]
        :param conv_state: Optional conv state tensor for this layer
        :param seq_ids: Sequence IDs for state indexing
        :param seq_lens: Sequence lengths for prefill
        :param is_prefill: Whether this is prefill or decode
        :param shortconv_layer_idx: Index within ShortConv layers (0-9)
        :returns: Tuple of (output tensor, updated conv state values)
        """
        bcx = self.in_proj(hidden_states)
        b, c, x = bcx.chunk(3, dim=-1)
        bx = b * x
        if is_prefill:
            output, new_states = self._prefill_conv(bx, c, seq_lens, conv_state, seq_ids)
        else:
            output, new_states = self._decode_conv(bx, c, conv_state, seq_ids)

        output = self.out_proj(output)
        return output, new_states


    def _prefill_conv(
        self,
        bx: Tensor,
        c: Tensor,
        seq_lens: Optional[list[int]],
        conv_state: Optional[Tensor] = None,
        seq_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute causal conv1d for prefill mode.

        :param bx: Gated input tensor of shape [num_tokens, conv_dim]
        :param c: Output gate tensor of shape [num_tokens, conv_dim]
        :param seq_lens: List of sequence lengths (chunk sizes)
        :param conv_state: Optional previous conv state for chunked prefill
        :param seq_ids: Sequence IDs for state indexing (conv slot indices)
        :returns: Tuple of (output tensor, final states per sequence)
        """
        if seq_lens is None:
            seq_lens = [bx.shape[0]]

        outputs = []
        states = []
        offset = 0
        state_len = self.kernel_size - 1

        for seq_idx, seq_len in enumerate(seq_lens):
            bx_seq = bx[offset : offset + seq_len]
            
            prev_state = None
            if conv_state is not None and seq_ids is not None:
                slot = seq_ids[seq_idx].item()
                prev_state = conv_state[slot]
                if prev_state.abs().sum() == 0:
                    prev_state = None

            if prev_state is not None:
                bx_with_state = torch.cat([prev_state, bx_seq], dim=0)
                bx_t = bx_with_state.transpose(0, 1).unsqueeze(0)
                conv_out = F.conv1d(
                    bx_t,
                    self.conv_weight,
                    padding=0,
                    groups=self.conv_dim,
                )
                conv_out = conv_out.squeeze(0).transpose(0, 1)
            else:
                bx_t = bx_seq.transpose(0, 1).unsqueeze(0)
                conv_out = F.conv1d(
                    bx_t,
                    self.conv_weight,
                    padding=self.kernel_size - 1,
                    groups=self.conv_dim,
                )
                conv_out = conv_out[:, :, : seq_len]
                conv_out = conv_out.squeeze(0).transpose(0, 1)

            c_seq = c[offset : offset + seq_len]
            y = c_seq * conv_out
            outputs.append(y)
            if seq_len >= state_len:
                state = bx_seq[-state_len:]
            else:
                if prev_state is not None:
                    combined = torch.cat([prev_state, bx_seq], dim=0)
                    state = combined[-state_len:]
                else:
                    pad_len = state_len - seq_len
                    state = F.pad(bx_seq, (0, 0, pad_len, 0))
            states.append(state)
            offset += seq_len

        return torch.cat(outputs, dim=0), torch.stack(states, dim=0)


    def _decode_conv(
        self,
        bx: Tensor,
        c: Tensor,
        conv_state: Tensor,
        seq_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute incremental conv1d for decode mode.

        :param bx: Gated input tensor of shape [batch_size, conv_dim]
        :param c: Output gate tensor of shape [batch_size, conv_dim]
        :param conv_state: Conv state of shape [max_seqs, kernel-1, conv_dim]
        :param seq_ids: Sequence IDs for state lookup
        :returns: Tuple of (output tensor, new state values)
        """
        batch_size = bx.shape[0]
        state_len = self.kernel_size - 1
        states = conv_state[seq_ids]
        full_input = torch.cat([states, bx.unsqueeze(1)], dim=1)
        weight_2d = self.conv_weight.squeeze(1)
        conv_out = torch.einsum("bkd,dk->bd", full_input, weight_2d)
        y = c * conv_out
        new_states = torch.cat([states[:, 1:], bx.unsqueeze(1)], dim=1)
        return y, new_states


class Lfm2MLP(nn.Module):
    """
    SwiGLU MLP layer for LFM2. Implements the gated linear unit with SiLU activation,
    using merged gate-up projection for efficiency.
    """

    def __init__(
        self,
        config: LfmConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize MLP with merged gate-up and down projections.

        :param config: LFM model configuration
        :param dtype: Data type for weight tensors
        """
        super().__init__()

        ff_dim = config.block_ff_dim
        if config.block_auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if config.block_ffn_dim_multiplier is not None:
                ff_dim = int(config.block_ffn_dim_multiplier * ff_dim)
            multiple_of = config.block_multiple_of
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.ff_dim = ff_dim

        self.w1 = nn.Linear(
            config.block_dim,
            2 * ff_dim,
            bias=False,
            dtype=dtype,
        )
        self.w2 = nn.Linear(
            ff_dim,
            config.block_dim,
            bias=False,
            dtype=dtype,
        )


    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Input tensor of shape [..., hidden_size]
        :returns: Output tensor of shape [..., hidden_size]
        """
        gate_up = self.w1(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.w2(x)
        return x


class Lfm2DecoderLayer(nn.Module):
    """
    Single decoder layer for LFM2 supporting both attention and ShortConv. Contains operator_norm, 
    operator (attention or ShortConv), ffn_norm, and MLP.
    """

    def __init__(
        self,
        config: LfmConfig,
        layer_idx: int,
        is_attention: bool,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize decoder layer with appropriate operator type.

        :param config: LFM model configuration
        :param layer_idx: Index of this layer in the model
        :param is_attention: Whether this layer uses attention vs ShortConv
        :param dtype: Data type for weight tensors
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention = is_attention
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype)
        if is_attention:
            self.operator = Lfm2Attention(config, layer_idx, dtype=dtype)
        else:
            self.operator = Lfm2ShortConv(config, layer_idx, dtype=dtype)

        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype)
        self.mlp = Lfm2MLP(config, dtype=dtype)


    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        residual: Optional[Tensor],
        kv_cache: Optional[Tensor] = None,
        conv_state: Optional[Tensor] = None,
        slot_mapping: Optional[Tensor] = None,
        context_lens: Optional[Tensor] = None,
        block_table: Optional[Tensor] = None,
        seq_ids: Optional[Tensor] = None,
        seq_lens: Optional[list[int]] = None,
        is_prefill: bool = True,
        attn_layer_idx: int = 0,
        shortconv_layer_idx: int = 0,
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """Execute decoder layer forward pass.

        :param hidden_states: Input tensor of shape [num_tokens, hidden_size]
        :param positions: Position indices of shape [num_tokens]
        :param residual: Residual tensor from previous layer
        :param kv_cache: Optional KV cache for attention layers
        :param conv_state: Optional conv state for ShortConv layers
        :param slot_mapping: Cache slot mapping for attention
        :param context_lens: Context lengths for decode attention
        :param block_table: Block table for paged attention
        :param seq_ids: Sequence IDs for conv state indexing
        :param seq_lens: Sequence lengths for prefill
        :param is_prefill: Whether this is prefill or decode
        :param attn_layer_idx: Index within attention layers
        :param shortconv_layer_idx: Index within ShortConv layers
        :returns: Tuple of (hidden_states, residual, optional new_conv_state)
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)

        new_conv_state = None

        if self.is_attention:
            hidden_states = self.operator(
                hidden_states,
                positions,
                kv_cache=kv_cache,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_table=block_table,
                seq_lens=seq_lens,
                is_prefill=is_prefill,
                attn_layer_idx=attn_layer_idx,
            )
        else:
            hidden_states, new_conv_state = self.operator(
                hidden_states,
                conv_state=conv_state,
                seq_ids=seq_ids,
                seq_lens=seq_lens,
                is_prefill=is_prefill,
                shortconv_layer_idx=shortconv_layer_idx,
            )

        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, new_conv_state
