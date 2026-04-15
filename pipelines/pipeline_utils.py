import torch
import inspect
import diffusers
import importlib
import dataclasses

from accelerate import load_checkpoint_and_dispatch
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

    # Same with huggingface from_pretrained
    pretrained_model_name_or_path: str

    # The forward handler full path that can be loaded by importlib
    handler_name: str

    checkpoint: dict[str, str] = dataclasses.field(default_factory=dict)

    # Modules that should be tuned. If specified, adapters will be attatched into them.
    adpt_tune_modules: list[str] = dataclasses.field(default_factory=list)

    # Modules that should be loaded. Useful when GPU memory is constrained as the
    # prompt_embeds can be prepared in advance.
    load_modules: list[str] = dataclasses.field(default_factory=list)

    # Tune orignal modules without adapter
    # This should be a dict[component_name, module_name]
    # Example: transformer: [attn.to_q, attn.to_k, attn.to_v]
    # means that these modules will be fully tuned.
    full_tune_modules: dict[str, list[str] | str] = dataclasses.field(default_factory=dict)


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


class TunePipelineManager:

    def __init__(
        self,
        pipe_configs: PipelineConfigs | OmegaConf,
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        **pipe_init_kwargs,
    ):
        self.weight_dtype = weight_dtype
        self.device = device

        if not isinstance(pipe_configs, PipelineConfigs):
            pipe_configs: PipelineConfigs = instantiate(pipe_configs)

        self.adpt_configs: AdapterConfigs = None
        self.pipe_configs = pipe_configs
        self.load_modules = self.pipe_configs.load_modules
        self.adpt_tune_modules = self.pipe_configs.adpt_tune_modules
        self.full_tune_modules = self.pipe_configs.full_tune_modules
        self.checkpoint = self.pipe_configs.checkpoint

        self.pipeline: DiffusionPipeline = self.init_pipeline(**pipe_init_kwargs)
        self.handler = ForwardHandler.from_configs(self.pipe_configs)

    def init_pipeline(self, **pipe_init_kwargs) -> DiffusionPipeline:
        r"""
        Initialize pipeline and return it.
        Note that this will `disable` all gradients in each modules and
        move them to target deivces with weight dtype.
        """
        if not set[str](self.pipe_configs.adpt_tune_modules).issubset(set[str](self.pipe_configs.load_modules)):
            raise ValueError(
                f"Tuning modules is not subset of loaded modules: "
                + f"tune_modules={self.pipe_configs.adpt_tune_modules}, "
                + f"load_modules={self.pipe_configs.load_modules}."
            )

        # Load modules that specified in load_modules,
        # avoiding load some no use modules to save memory.
        module_dict = {
            m: AutoModel.from_pretrained(
                self.pipe_configs.pretrained_model_name_or_path,
                subfolder=m,
                torch_dtype=self.weight_dtype,
            )
            for m in self.pipe_configs.load_modules
        }
        for n, m in module_dict.items():
            if not isinstance(m, torch.nn.Module):
                continue
            for p in m.parameters():
                p.requires_grad_(False)
            if self.checkpoint and m in self.checkpoint:
                load_checkpoint_and_dispatch(m, self.checkpoint[m], device_map="auto")
            m.to(device=self.device, dtype=self.weight_dtype)
            logger.info(f"{n} to {self.device} with {self.weight_dtype}, no grad.")

        # Specify the pipeline, get the class
        model_path = self.pipe_configs.pretrained_model_name_or_path
        config_dict = DiffusionPipeline.load_config(model_path)
        pipeline_cls_name = config_dict.get("_class_name", "DiffusionPipeline")
        pipeline_cls = getattr(diffusers, pipeline_cls_name)

        # Specify the pipeline args and fill them with loaded modules and default kwargs
        # For diffusers pipelines, only take modules as kwargs
        # For other not module kwargs, we set None by default.
        pipe_signature = inspect.signature(pipeline_cls.__init__)
        pipe_kwargs = {}
        for name, param in pipe_signature.parameters.items():
            if name == "self":
                continue
            if name in module_dict:
                pipe_kwargs[name] = module_dict[name]
            elif name in pipe_init_kwargs:
                pipe_kwargs[name] = pipe_init_kwargs[name]

        pipeline = pipeline_cls(**pipe_kwargs)
        pipeline.to(device=self.device, dtype=self.weight_dtype)
        return pipeline

    def get_module(self, name: str) -> torch.nn.Module | None:
        return self.pipeline.components.get(name, None)

    def add_adapter(
        self,
        adpt_configs: AdapterConfigs,
        tune_modules: list[str] | None = None,
        requires_grad: bool = False,
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> DiffusionPipeline:
        r"""
        Insert adapter into tune_modules.

        :param adpt_checkpoint: Adapter checkpoint that ends with "pth" or "safetensors".
        """
        adpt_checkpoint = adpt_configs.checkpoint
        tune_modules = tune_modules or self.pipe_configs.adpt_tune_modules
        weight_dtype = weight_dtype or self.weight_dtype
        device = device or self.device

        if not set[str](tune_modules).issubset(set[str](self.pipe_configs.load_modules)):
            raise ValueError(
                f"Tuning modules is not subset of loaded modules: "
                + f"tune_modules={tune_modules}, "
                + f"load_modules={self.pipe_configs.load_modules}."
            )
        logger.info(f"Following modules will be tuned: {tune_modules}. Requires grad: {requires_grad}.")

        # Load adpt checkpoint if founded and valid
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
                    if _ckpt_type == "pth"
                    else load_file(adpt_checkpoint, device=device)
                )
            else:
                logger.warning(
                    f"Adapter checkpoint unrecognized: {adpt_checkpoint}. "
                    + f"Supported file type: {_supported_file_types}"
                )

        # The adapter loaded workflow is
        # register adapter manager -> add an adapter instance ->
        # enable the gradients if required -> activate target
        # adapter (if there are multiple adapter, this is useful)
        for module_name in self.pipeline.components:
            if module_name not in tune_modules:
                continue
            logger.info(f"Add adapter in: {module_name}.")
            module = self.get_module(module_name)
            register_adapter(module)
            add_adapter(module, adpt_configs, overwrite=False)
            if requires_grad:
                logger.info(f"Adapter in {module_name} enable gradient.")
                enable_adapter(module, adpt_configs.adapter_name)
            activate_adapter(module, adpt_configs.adapter_name)

            if adpt_state_dicts:
                adpt_pn = set()
                module_pn = set()
                logger.info(f"Load adapter checkpoints to {module_name}.")
                for n in adpt_state_dicts:
                    if module_name in n and adpt_configs.adapter_name in n:
                        adpt_pn.add(n)
                for n, _ in module.named_parameters():
                    module_pn.add(n)
                loaded_pn = adpt_pn.intersection(module_pn)
                logger.info(
                    f"Adapter params: {len(adpt_pn)} | "
                    + f"Module params: {len(module_pn)} | "
                    + f"Loaded (intersection) params: {len(loaded_pn)}"
                )
                module.load_state_dict(adpt_state_dicts, strict=False)
        self.pipeline.to(device=device, dtype=weight_dtype)
        self.adpt_configs = adpt_configs
        return self.pipeline

    def enable_full_tune_modules(
        self,
        full_tune_modules: dict[str, list[str] | str] = None,
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> DiffusionPipeline:
        r"""
        Enable gradient for full tune modules.
        We use a naive way to find all trainable params, that is to check if the
        names in full_tune_modules is contained by the module_names of pipelines.

        :param full_tune_modules: The format is the dict of component name to path
            to module, splited by dots. Note that it is different from load_modules
            and tune_modules, which is the component names of pipeline.
        """
        if not full_tune_modules:
            return self.pipeline
        full_tune_modules = full_tune_modules or self.full_tune_modules
        weight_dtype = weight_dtype or self.weight_dtype
        device = device or self.device

        for ftm_name in full_tune_modules:
            if ftm_name not in self.load_modules:
                logger.warning(
                    f"Full tune module {ftm_name} is not loaded. "
                    + "You should add it into load_modules if you want to tune it, "
                    + "otherwise it will not be full tuned."
                )
                continue
            enabled_module_names = []
            for n, m in self.get_module(ftm_name).named_modules():
                if not any(p in n for p in full_tune_modules[ftm_name]):
                    continue
                m.requires_grad_(True)
                enabled_module_names.append(n)
            logger.info(f"Found {len(enabled_module_names)} modules in component {ftm_name} enable gradient.")
        return self.pipeline

    def __call__(self, batch: dict[str, Any]) -> ForwardOutputs:
        outputs = self.handler.forward_step(self.pipeline, batch)
        return outputs
