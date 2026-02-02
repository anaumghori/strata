from typing import Callable
import torch
from torch import nn

from strata.models.lfm.model import build_lfm_model


MODEL_REGISTRY: dict[str, Callable[[str, torch.dtype], nn.Module]] = {
    "lfm": build_lfm_model,
    "lfm2": build_lfm_model,
    "liquidai": build_lfm_model,
}


def get_model_builder(model_name: str) -> Callable[[str, torch.dtype], nn.Module]:
    """Matches model names/paths against registered patterns. 

    :param model_name: Name or path of the model
    :returns: Builder function that takes (path, dtype) and returns model
    """
    model_key = model_name.lower()

    for key, builder in MODEL_REGISTRY.items():
        if key in model_key:
            return builder

    return build_lfm_model


def build_model(model_path: str, dtype: torch.dtype = torch.float32) -> nn.Module:
    """
    :param model_path: Path to the model directory
    :param dtype: Data type for model weights
    :returns: Initialized and loaded model
    """
    builder = get_model_builder(model_path)
    return builder(model_path, dtype)
