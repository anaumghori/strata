from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path


@dataclass
class LfmConfig:
    """
    Configuration for LFM2 model architecture. Parses HuggingFace config.json to 
    extract model dimensions, layer types, and normalization parameters.
    """

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    norm_eps: float
    rope_theta: float
    conv_dim: int
    conv_kernel_size: int
    tie_word_embeddings: bool
    layer_types: list[str]
    block_dim: int
    block_ff_dim: int
    block_multiple_of: int
    block_auto_adjust_ff_dim: bool
    block_ffn_dim_multiplier: Optional[float]


def load_lfm_config(model_path: str) -> LfmConfig:
    """
    :param model_path: Path to model directory containing config.json
    :returns: LfmConfig populated from the configuration file
    """
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    hidden_size = cfg["hidden_size"]
    num_hidden_layers = cfg["num_hidden_layers"]
    num_attention_heads = cfg["num_attention_heads"]
    num_key_value_heads = cfg.get("num_key_value_heads", num_attention_heads)

    head_dim = cfg.get("head_dim")
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads

    layer_types = cfg.get("layer_types")
    if layer_types is None:
        layer_types = _infer_layer_types(cfg, num_hidden_layers)

    conv_dim = cfg.get("conv_dim", hidden_size)
    block_dim = cfg.get("block_dim", hidden_size)

    return LfmConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=cfg.get("intermediate_size", hidden_size * 4),
        max_position_embeddings=cfg.get("max_position_embeddings", 32768),
        norm_eps=cfg.get("norm_eps", cfg.get("rms_norm_eps", 1e-5)),
        rope_theta=cfg.get("rope_theta", 1000000.0),
        conv_dim=conv_dim,
        conv_kernel_size=cfg.get("conv_L_cache", cfg.get("conv_kernel_size", 3)),
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        layer_types=layer_types,
        block_dim=block_dim,
        block_ff_dim=cfg.get("block_ff_dim", hidden_size * 4),
        block_multiple_of=cfg.get("block_multiple_of", 256),
        block_auto_adjust_ff_dim=cfg.get("block_auto_adjust_ff_dim", True),
        block_ffn_dim_multiplier=cfg.get("block_ffn_dim_multiplier"),
    )


def _infer_layer_types(cfg: dict, num_hidden_layers: int) -> list[str]:
    """Infer layer types from full_attn_idxs when layer_types is not provided.

    :param cfg: Raw config dictionary
    :param num_hidden_layers: Number of layers in the model
    :returns: List of layer type strings
    """
    full_attn_idxs = cfg.get("full_attn_idxs")
    if full_attn_idxs is None:
        raise ValueError(
            "Cannot determine layer types: config.json must contain "
            "'layer_types' or 'full_attn_idxs' field."
        )

    attn_set = set(full_attn_idxs)
    return [
        "full_attention" if i in attn_set else "short_conv"
        for i in range(num_hidden_layers)
    ]


def get_attention_layer_indices(config: LfmConfig) -> list[int]:
    """
    :param config: LFM model configuration
    :returns: List of layer indices that use attention
    """
    return [
        i for i, layer_type in enumerate(config.layer_types)
        if _is_attention_layer(layer_type)
    ]


def get_shortconv_layer_indices(config: LfmConfig) -> list[int]:
    """
    :param config: LFM model configuration
    :returns: List of layer indices that use ShortConv
    """
    return [
        i for i, layer_type in enumerate(config.layer_types)
        if not _is_attention_layer(layer_type)
    ]


def _is_attention_layer(layer_type: str) -> bool:
    """
    :param layer_type: Layer type string from config
    :returns: True if this is an attention layer
    """
    return layer_type == "full_attention"
