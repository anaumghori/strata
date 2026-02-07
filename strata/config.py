import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class StrataConfig:
    """Configuration for Strata inference engine."""

    model_path: str
    max_num_seqs: int = 32
    max_context_length: int = 16384
    block_size: int = 16
    memory_utilization: float = 0.9
    max_num_batched_tokens: int = 4096
    prefill_chunk_size: int = 256
    prefill_fairness_chunks: int = 4
    enable_prefix_caching: bool = True
    dtype: str = "auto"
    num_threads: Optional[int] = None
    use_compile: bool = False
    default_temperature: float = 1.0
    default_max_tokens: int = 256


@dataclass
class SamplingParams:
    """Parameters controlling text generation sampling behavior."""

    temperature: float = 1.0
    max_tokens: int = 256
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[list[str]] = None
    ignore_eos: bool = False


_STR_TO_TORCH_DTYPE = {
    "float16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "float": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _is_bf16_supported() -> bool:
    """Check if bfloat16 is supported on current hardware."""
    try:
        test_tensor = torch.zeros(1, dtype=torch.bfloat16)
        _ = test_tensor + test_tensor
        return True
    except (RuntimeError, TypeError):
        return False


def _get_model_config_dtype(model_path: str) -> torch.dtype | None:
    """Read torch_dtype from model's config.json file"""
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        torch_dtype_str = config.get("torch_dtype")
        if torch_dtype_str is None:
            return None

        return _STR_TO_TORCH_DTYPE.get(torch_dtype_str.lower())
    except (json.JSONDecodeError, OSError, AttributeError):
        return None


def _get_hardware_preferred_dtype() -> torch.dtype:
    """Determine the best dtype supported by current hardware"""
    if _is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def get_dtype(dtype_str: str, model_path: str | None = None) -> torch.dtype:
    """Determine the optimal dtype based on configuration, model, and hardware"""
    dtype_str_lower = dtype_str.lower()
    if dtype_str_lower != "auto":
        if dtype_str_lower in _STR_TO_TORCH_DTYPE:
            return _STR_TO_TORCH_DTYPE[dtype_str_lower]
        return torch.float32

    if model_path is not None:
        model_dtype = _get_model_config_dtype(model_path)
        if model_dtype is not None:
            if model_dtype == torch.bfloat16 and not _is_bf16_supported():
                return torch.float32
            return model_dtype

    return _get_hardware_preferred_dtype()
