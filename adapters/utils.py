import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from torch import Tensor
from typing import List, Optional, Any, Dict

from loguru import logger


DEFAULT_RANK = 16
DEFAULT_NAME = "default"


@dataclass
class AdapterConfigs:

    rank: int = DEFAULT_RANK
    alpha: Optional[int] = None
    adapter_name: str = DEFAULT_NAME
    target_modules: list[str] = field(default_factory=[])
    bias: bool = False


class AdapterManager(nn.Module):
    r"""
    Adapter provides basic function of low rank finetune.
    It includes parameter dicts like `lora_A` and `lora_B`, which are same with
    the naming rule of `peft`. Moreover, it uses additional `bias_A` and `bias_B`
    to adapt to full-zero inputs sometimes.

    As we can see, this class just a meta-manager for other adapters with same
    module structures but different configs (such as different ranks). This design
    simplies the process of adding adapters.
    """

    def __init__(self, base_model: nn.Linear):
        super().__init__()

        self.base_model = base_model
        self.in_features = base_model.in_features
        self.out_features = base_model.out_features

        self.lora_A = nn.ParameterDict()  # (rank, dim)
        self.bias_A = nn.ParameterDict()  # (rank, )
        self.lora_B = nn.ParameterDict()  # (dim, rank)
        self.bias_B = nn.ParameterDict()  # (dim, )

        # adapter info stores basic information about the added adapters
        # keys:
        #   name: adapter name
        #       is_enable: identify whether this adapter is enabled
        #       configs: same keys with AdapterConfigs
        self.adapter_info: Dict[str, Any] = {}

        # Record the activated adapter name
        self.active_adapter: str = ""

    def forward(self, x: Tensor):
        out = self.base_model(x)

        if self.active_adapter and self.adapter_info[self.active_adapter]["is_enable"]:
            configs = self.adapter_info[self.active_adapter]["configs"]
            rank = configs.get("rank", DEFAULT_RANK)
            alpha = configs.get("alpha") or rank
            scale = alpha / rank

            A: torch.Tensor = self.lora_A[self.active_adapter]
            bias_A = self.bias_A[self.active_adapter] if self.active_adapter in self.bias_A else None
            B = self.lora_B[self.active_adapter]
            bias_B = self.bias_B[self.active_adapter] if self.active_adapter in self.bias_B else None

            # Use F.linear to avoid explicit transpose matmul and large temporary tensor.
            # IMPORTANT: do not scale x first (i.e. `scale * x @ A.T`), which creates an
            # extra tensor with the same shape as x and can trigger OOM.
            ada_out = F.linear(x, A, bias_A)
            ada_out = F.linear(ada_out, B, bias_B)
            out = out + ada_out * scale
        return out


def register_adapter(model: nn.Module):
    r"""
    Register finetune adapter (the class Adapter) on all model linear layers.
    Given the adapter configurations, only target_modules will be used,
    this method will find all matched modules and replace them with adapter
    layers.

    Note that this method will NOT add any trainable parameters into
    the model, it just register some ParameterDict/ModuleDict entrances
    for later usage.

    If you want to *add* an adapter instance, use `add_adapter` instead.
    """
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        splited_name = name.split(".")
        parent_name = ".".join(splited_name[:-1])
        parent_node = model.get_submodule(parent_name)
        target_name = splited_name[-1]

        adapter = AdapterManager(module)
        setattr(parent_node, target_name, adapter)


def add_adapter(model: nn.Module, configs: AdapterConfigs, overwrite: bool = False):
    r"""
    Add finetune adapter (just add parameters to dict of class Adapter) in model.
    """
    target_modules: List[str] = configs.target_modules
    adapter_name: str = configs.adapter_name
    rank: int = configs.rank

    for name, module in model.named_modules():
        if not any(n in name.split(".") for n in target_modules):
            print(f"{name} not add adapter.")
            continue
        if not isinstance(module, AdapterManager):
            continue

        if adapter_name in module.adapter_info and not overwrite:
            logger.error(
                f"{adapter_name} is already defined in modeul {name}. If you want to overwrite the existing adapter with new one, set `overwrite=True`."
            )
            raise KeyError(
                f"{adapter_name} is already defined in modeul {name}. If you want to overwrite the existing adapter with new one, set `overwrite=True`."
            )

        in_features = module.in_features
        out_features = module.out_features

        # Initialization
        # In fact, more initialization methods can be provided via method args
        # and macro definitions. Here we use the normal gaussian and zero init method
        # by default, which is also used in `peft`.
        module.lora_A[adapter_name] = nn.Parameter(torch.empty(rank, in_features), requires_grad=False)
        module.lora_B[adapter_name] = nn.Parameter(torch.empty(out_features, rank), requires_grad=False)
        nn.init.normal_(module.lora_A[adapter_name], mean=0, std=0.02)
        nn.init.zeros_(module.lora_B[adapter_name])

        if configs.bias:
            module.bias_A[adapter_name] = nn.Parameter(torch.empty(rank), requires_grad=False)
            init_bias_A = 1 / math.sqrt(rank)
            nn.init.uniform_(module.bias_A[adapter_name], -init_bias_A, init_bias_A)

            module.bias_B[adapter_name] = nn.Parameter(torch.empty(out_features), requires_grad=False)
            nn.init.zeros_(module.bias_B[adapter_name])

        # Update the config of this module's adapter
        module.adapter_info[adapter_name] = {}
        module.adapter_info[adapter_name]["configs"] = {
            "rank": rank,
            "alpha": configs.alpha,
            "target_modules": target_modules,
            "adapter_name": adapter_name,
            "bias": configs.bias,
        }
        module.adapter_info[adapter_name]["is_enable"] = False
        module.adapter_info[adapter_name]["is_active"] = False


def enable_adapter(model: nn.Module, name: str):
    r"""
    Enable the gradient to True for adapter specific by name.
    """
    for _, module in model.named_modules():
        if not isinstance(module, AdapterManager):
            continue
        if not name in module.adapter_info:
            continue

        module.lora_A[name].requires_grad_(True)
        module.lora_B[name].requires_grad_(True)

        if module.adapter_info[name]["configs"]["bias"]:
            module.bias_A[name].requires_grad_(True)
            module.bias_B[name].requires_grad_(True)

        module.adapter_info[name]["is_enable"] = True


def disable_adapter(model: nn.Module, name: str):
    r"""
    disable the gradient to True for adapter specific by name.
    """
    for _, module in model.named_modules():
        if not isinstance(module, AdapterManager):
            continue
        if not name in module.adapter_info:
            continue
        module.lora_A[name].requires_grad_(False)
        module.lora_B[name].requires_grad_(False)

        if module.adapter_info[name]["configs"]["bias"]:
            module.bias_A[name].requires_grad_(False)
            module.bias_B[name].requires_grad_(False)

        module.adapter_info[name]["is_enable"] = False


def activate_adapter(model: nn.Module, name: str):
    for _, module in model.named_modules():
        if not isinstance(module, AdapterManager):
            continue
        if not name in module.adapter_info:
            continue

        module.active_adapter = name
        module.adapter_info[name]["is_active"] = True


def deactivate_adapter(model: nn.Module, name: str):
    for _, module in model.named_modules():
        if not isinstance(module, AdapterManager):
            continue
        if not name in module.adapter_info:
            continue

        module.active_adapter = ""
        module.adapter_info[name]["is_active"] = False
