import json
import diffusers
import torch
import importlib
import inspect
import dataclasses

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.auto_model import AutoModel
from functools import partial
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from pathlib import Path
from safetensors.torch import load_file
from typing import Any, Callable

from adapters.utils import AdapterConfigs
from adapters.utils import register_adapter, add_adapter, enable_adapter, activate_adapter


@dataclasses.dataclass
class PipelineConfigs:

    pretrained_model_name_or_path: str
    handler_name: str

    # Modules that should be tuned. If specified, adapters will be attatched into them.
    tune_modules: list[str] = dataclasses.field(default_factory=lambda: [])

    # Modules that should be loaded. Useful when GPU memory is constrained as the
    # prompt_embeds can be prepared in advance.
    load_modules: list[str] = dataclasses.field(default_factory=lambda: [])


@dataclasses.dataclass
class ConditionOutputs:

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None = None
    others: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LatentOutputs:

    noise_latents: torch.Tensor
    target_latents: torch.Tensor
    condition_latents: torch.Tensor | None = None
    others: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SampleOutputs:

    noisy_latents: torch.Tensor
    timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None


@dataclasses.dataclass
class DenoiseOutputs:

    predictions: torch.Tensor


@dataclasses.dataclass
class CriterionOutputs:

    loss: torch.Tensor


@dataclasses.dataclass
class ForwardOutputs:

    loss: torch.Tensor
    predictions: torch.Tensor | None = None


class ForwardHandler:

    def __init__(self):
        pass

    @staticmethod
    def fn_auto_fill(fn: Callable, batch: dict[str, Any], **kwargs) -> Callable:
        r"""
        Fill the args in the given function automatically by batch keys and other kwargs.

        This is useful when the batch share the same keys with the given function, no need
        to rewrite the function again.

        Return
            A callable partial function with filled keyword arguments.
        """
        fn_signature = inspect.signature(fn)
        fn_kwargs = {}
        fn_keys = set(fn_signature.parameters.keys())
        valid_batch_keys = set(batch.keys()).intersection(fn_keys)
        valid_other_keys = set(kwargs.keys()).intersection(fn_keys)
        for k in valid_batch_keys:
            fn_kwargs[k] = batch[k]
        for k in valid_other_keys:
            fn_kwargs[k] = kwargs[k]
        return partial(fn, **fn_kwargs)

    @staticmethod
    def from_configs(pipe_configs: PipelineConfigs) -> "ForwardHandler":
        handler_path = pipe_configs.handler_name
        handler_module = importlib.import_module(".".join(handler_path.split(".")[:-1]))
        handler: "ForwardHandler" = getattr(handler_module, handler_path.split(".")[-1])()
        return handler

    @staticmethod
    def check_inputs(pipeline: DiffusionPipeline, batch: dict[str, Any], **kwargs):
        logger.warning(
            f"{inspect.currentframe().f_code.co_name} have not been implemented yet. "
            + f"No checking on input batch, this may cause inconsistency or error later."
        )

    @staticmethod
    def encode_conditions(pipeline: DiffusionPipeline, batch: dict[str, Any], **kwargs) -> ConditionOutputs:
        raise NotImplementedError(f"{inspect.currentframe().f_code.co_name} have not been implemented yet.")

    @staticmethod
    def encode_latents(pipeline: DiffusionPipeline, batch: dict[str, Any], **kwargs) -> LatentOutputs:
        raise NotImplementedError(f"{inspect.currentframe().f_code.co_name} have not been implemented yet.")

    @staticmethod
    def sample_latents(pipeline: DiffusionPipeline, batch: dict[str, Any], **kwargs) -> SampleOutputs:
        raise NotImplementedError(f"{inspect.currentframe().f_code.co_name} have not been implemented yet.")

    @staticmethod
    def denoise_forward(pipeline: DiffusionPipeline, batch: dict[str, Any], **kwargs) -> DenoiseOutputs:
        raise NotImplementedError(f"{inspect.currentframe().f_code.co_name} have not been implemented yet.")

    @staticmethod
    def criterion_fn(pipeline: DiffusionPipeline, batch: dict[str, Any], **kwargs) -> CriterionOutputs:
        raise NotImplementedError(f"{inspect.currentframe().f_code.co_name} have not been implemented yet.")

    @staticmethod
    def forward_step(pipeline: DiffusionPipeline, batch: dict[str, Any], **kwargs) -> ForwardOutputs:
        r"""
        The entrance of training. Must be implemented to describe how to compose above method as there might
        be some specified parameters for above methods that do not be included in batch.
        """
        raise NotImplementedError(f"{inspect.currentframe().f_code.co_name} have not been implemented yet.")


class TunePipeline:

    def __init__(
        self,
        pipe_configs: PipelineConfigs | OmegaConf,
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        self.weight_dtype = weight_dtype
        self.device = device

        if not isinstance(pipe_configs, PipelineConfigs):
            pipe_configs: PipelineConfigs = instantiate(pipe_configs)

        self.adpt_configs = {}
        self.pipe_configs = pipe_configs
        self.tune_modules = self.pipe_configs.tune_modules

        self.pipeline = self.init_pipeline()
        self.handler = ForwardHandler.from_configs(self.pipe_configs)

    def init_pipeline(self) -> DiffusionPipeline:
        if not set[str](self.pipe_configs.tune_modules).issubset(set[str](self.pipe_configs.load_modules)):
            raise ValueError(
                f"Tuning modules is not subset of loaded modules: tune_modules={self.pipe_configs.tune_modules}, load_modules={self.pipe_configs.load_modules}."
            )

        module_dict = {
            m: AutoModel.from_pretrained(
                self.pipe_configs.pretrained_model_name_or_path, subfolder=m, torch_dtype=self.weight_dtype
            ).to(self.device)
            for m in self.pipe_configs.load_modules
        }

        # Specify the pipeline
        model_index_file = Path(self.pipe_configs.pretrained_model_name_or_path) / "model_index.json"
        with open(model_index_file, "r") as f:
            pipeline_cls_name = json.load(f).get("_class_name", "DiffusionPipeline")
        pipeline_cls = getattr(diffusers, pipeline_cls_name)

        # Specify the pipeline args
        pipe_signature = inspect.signature(pipeline_cls.__init__)
        pipe_kwargs = {}
        for name, param in pipe_signature.parameters.items():
            if name == "self":
                continue
            if name in module_dict:
                pipe_kwargs[name] = module_dict[name]
            elif param.default is inspect.Parameter.empty:
                pipe_kwargs[name] = None
            else:
                continue

        pipeline = pipeline_cls(**pipe_kwargs)
        pipeline.to(device=self.device, dtype=self.weight_dtype)

        return pipeline

    def add_adapter(
        self,
        adpt_configs: AdapterConfigs,
        tune_modules: list[str] | None = None,
        requires_grad: bool = False,
        adpt_checkpoint: str | Path | None = None,
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> DiffusionPipeline:
        tune_modules = tune_modules or self.pipe_configs.tune_modules
        weight_dtype = weight_dtype or self.weight_dtype
        device = device or self.device

        if not set[str](tune_modules).issubset(set[str](self.pipe_configs.load_modules)):
            raise ValueError(
                f"Tuning modules is not subset of loaded modules: "
                + f"tune_modules={tune_modules}, "
                + f"load_modules={self.pipe_configs.load_modules}."
            )
        logger.info(f"Following modules will be tuned: {tune_modules}. Requires grad: {requires_grad}.")

        _ckpt_type = "safetensor"
        _supported_file_types = (".safetensors", ".pth")
        adpt_state_dicts = {}
        if adpt_checkpoint:
            if isinstance(adpt_checkpoint, str):
                adpt_checkpoint = Path(adpt_checkpoint)
            if adpt_checkpoint.exists() and adpt_checkpoint.suffix in _supported_file_types:
                logger.info(f"Adapter checkpoint is found in {adpt_checkpoint}.")
                _ckpt_type = adpt_checkpoint.suffix.split(".")[-1]
                adpt_state_dicts: dict[str, torch.Tensor] = (
                    torch.load(adpt_checkpoint, map_location=device, weights_only=True)
                    if _ckpt_type == ".pth"
                    else load_file(adpt_checkpoint, device=device)
                )
            else:
                logger.warning(
                    f"Adapter checkpoint unrecognized: {adpt_checkpoint}. "
                    + f"Supported file type: {_supported_file_types}"
                )

        for module in self.pipeline.components:
            if module not in tune_modules:
                continue
            register_adapter(self.pipeline.components[module])
            add_adapter(self.pipeline.components[module], adpt_configs, overwrite=False)
            if requires_grad:
                enable_adapter(self.pipeline.components[module], adpt_configs.adapter_name)
            activate_adapter(self.pipeline.components[module], adpt_configs.adapter_name)

        self.pipeline.to(device=device, dtype=weight_dtype)
        for module in self.pipeline.components:
            self.pipeline.components[module].load_state_dicts(adpt_state_dicts, strict=False)
        
        self.adpt_configs[adpt_configs.adapter_name] = adpt_configs
        return self.pipeline

    def __call__(self, batch: dict[str, Any]) -> ForwardOutputs:
        outputs = self.handler.forward_step(self.pipeline, batch)
        return outputs
