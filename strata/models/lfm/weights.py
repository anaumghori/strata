from glob import glob
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open

from strata.models.lfm.config import LfmConfig, _is_attention_layer


QKV_STACKED_MAPPING = {
    "self_attn.q_proj.weight": ("operator.qkv_proj.weight", "q"),
    "self_attn.k_proj.weight": ("operator.qkv_proj.weight", "k"),
    "self_attn.v_proj.weight": ("operator.qkv_proj.weight", "v"),
}

GATE_UP_STACKED_MAPPING = {
    "feed_forward.w1.weight": ("mlp.w1.weight", 0),
    "feed_forward.w3.weight": ("mlp.w1.weight", 1),
}


def load_lfm_weights(
    model: nn.Module,
    model_path: str,
    config: LfmConfig,
) -> None:
    """
    Weight shapes are derived from the loaded tensors and target model parameters.

    :param model: The LFM2 model to load weights into
    :param model_path: Path to directory containing safetensors files
    :param config: LFM configuration for layer type mapping
    :returns: None
    """
    weight_files = glob(str(Path(model_path) / "*.safetensors"))
    params_dict = dict(model.named_parameters())
    attn_indices = set()
    conv_indices = set()
    for i, layer_type in enumerate(config.layer_types):
        if _is_attention_layer(layer_type):
            attn_indices.add(i)
        else:
            conv_indices.add(i)

    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for hf_name in f.keys():
                loaded_weight = f.get_tensor(hf_name)
                _load_weight(
                    params_dict,
                    hf_name,
                    loaded_weight,
                    attn_indices,
                    conv_indices,
                )


def _load_weight(
    params_dict: dict,
    hf_name: str,
    loaded_weight: torch.Tensor,
    attn_indices: set,
    conv_indices: set,
) -> None:
    """Load a single weight tensor into the model.

    :param params_dict: Dictionary of model parameters
    :param hf_name: HuggingFace weight name
    :param loaded_weight: Weight tensor to load
    :param attn_indices: Set of attention layer indices
    :param conv_indices: Set of ShortConv layer indices
    :returns: None
    """
    if hf_name.startswith("model."):
        name = hf_name[6:]
    else:
        name = hf_name

    if not name.startswith("layers."):
        param_name = _map_simple_weight(name)
        if param_name and param_name in params_dict:
            params_dict[param_name].data.copy_(loaded_weight)
        return

    parts = name.split(".")
    layer_idx = int(parts[1])
    rest = ".".join(parts[2:])

    is_attention = layer_idx in attn_indices

    if is_attention and rest in QKV_STACKED_MAPPING:
        target_suffix, shard_id = QKV_STACKED_MAPPING[rest]
        param_name = f"layers.{layer_idx}.{target_suffix}"
        if param_name in params_dict:
            _load_qkv_shard(params_dict[param_name], loaded_weight, shard_id)
        return

    if rest in GATE_UP_STACKED_MAPPING:
        target_suffix, shard_id = GATE_UP_STACKED_MAPPING[rest]
        param_name = f"layers.{layer_idx}.{target_suffix}"
        if param_name in params_dict:
            _load_gate_up_shard(params_dict[param_name], loaded_weight, shard_id)
        return

    if not is_attention and rest == "conv.conv.weight":
        param_name = f"layers.{layer_idx}.operator.conv_weight"
        if param_name in params_dict:
            _load_conv_weight(params_dict[param_name], loaded_weight)
        return

    if is_attention:
        param_name = _map_attention_weight(layer_idx, rest)
    else:
        param_name = _map_shortconv_weight(layer_idx, rest)

    if param_name and param_name in params_dict:
        params_dict[param_name].data.copy_(loaded_weight)


def _map_simple_weight(name: str) -> str | None:
    """Map non-layer weight names.

    :param name: Weight name without model prefix
    :returns: Mapped parameter name or None
    """
    mapping = {
        "embed_tokens.weight": "embed_tokens.weight",
        "embedding_norm.weight": "embedding_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    return mapping.get(name)


def _map_attention_weight(layer_idx: int, rest: str) -> str | None:
    """
    :param layer_idx: Layer index in the model
    :param rest: Remaining weight path after layer index
    :returns: Mapped parameter name
    """
    mapping = {
        "self_attn.out_proj.weight": "operator.out_proj.weight",
        "self_attn.q_layernorm.weight": "operator.q_layernorm.weight",
        "self_attn.k_layernorm.weight": "operator.k_layernorm.weight",
        "operator_norm.weight": "operator_norm.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "feed_forward.w2.weight": "mlp.w2.weight",
    }
    if rest in mapping:
        return f"layers.{layer_idx}.{mapping[rest]}"
    return None


def _map_shortconv_weight(layer_idx: int, rest: str) -> str | None:
    """
    :param layer_idx: Layer index in the model
    :param rest: Remaining weight path after layer index
    :returns: Mapped parameter name
    """
    mapping = {
        "conv.in_proj.weight": "operator.in_proj.weight",
        "conv.out_proj.weight": "operator.out_proj.weight",
        "operator_norm.weight": "operator_norm.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "feed_forward.w2.weight": "mlp.w2.weight",
    }
    if rest in mapping:
        return f"layers.{layer_idx}.{mapping[rest]}"
    return None


def _load_conv_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
) -> None:
    """
    :param param: Target conv_weight parameter of shape [conv_dim, 1, kernel_size]
    :param loaded_weight: HF weight of shape [conv_dim, kernel_size]
    :returns: None
    """
    if loaded_weight.dim() == 2:
        loaded_weight = loaded_weight.unsqueeze(1)
    param.data.copy_(loaded_weight)


def _load_qkv_shard(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
) -> None:
    """Load Q, K, or V weight shard into merged qkv_proj parameter.

    :param param: Target qkv_proj parameter
    :param loaded_weight: Weight tensor to load
    :param shard_id: Shard identifier ('q', 'k', or 'v')
    :returns: None
    """
    param_data = param.data
    output_dim = param_data.shape[0]
    loaded_size = loaded_weight.shape[0]

    if shard_id == "q":
        param_data[:loaded_size].copy_(loaded_weight)
    elif shard_id == "k":
        kv_size = loaded_size
        q_size = output_dim - 2 * kv_size
        param_data[q_size : q_size + kv_size].copy_(loaded_weight)
    elif shard_id == "v":
        kv_size = loaded_size
        q_size = output_dim - 2 * kv_size
        param_data[q_size + kv_size :].copy_(loaded_weight)


def _load_gate_up_shard(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: int,
) -> None:
    """Load gate or up weight shard into merged w1 parameter.

    :param param: Target w1 parameter (merged gate+up)
    :param loaded_weight: Weight tensor to load
    :param shard_id: Shard identifier (0 for gate/w1, 1 for up/w3)
    :returns: None
    """
    param_data = param.data
    shard_size = param_data.shape[0] // 2
    offset = shard_id * shard_size
    param_data[offset : offset + shard_size].copy_(loaded_weight)
