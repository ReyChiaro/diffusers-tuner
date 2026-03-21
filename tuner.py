import os
import math
import torch

from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from hydra.utils import instantiate
from omegaconf import OmegaConf
from prodigyopt import Prodigy
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from typing import Optional, Dict

from adapters.utils import (
    AdapterConfigs,
    register_adapter,
    add_adapter,
    enable_adapter,
    activate_adapter,
    disable_adapter,
    deactivate_adapter,
)

# TODO: Convert to arbitrary pipeline loading
from pipelines.pipeline_qwenimage_edit_plus import tuning_step
from pipelines.pipeline_qwenimage_edit_plus import init_pipeline


class Tuner:

    @staticmethod
    def freeze_parameters(module: torch.nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    @staticmethod
    def summarize_model(model):
        # Calculate parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        # Calculate size on disk/memory (assuming float32 = 4 bytes)
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_ratio(%)": trainable_params / total_params,
            "model_size_mb(MB)": size_all_mb,
        }

    @staticmethod
    def init_pipeline(pipe_config: OmegaConf, weight_dtype: torch.dtype, device: torch.device, **kwargs) -> DiffusionPipeline:
        return init_pipeline(pipe_config, weight_dtype, device, **kwargs)

    @staticmethod
    def finetune(
        accelerator: Accelerator,
        pipe_config: OmegaConf,
        adpt_config: OmegaConf,
        loss_config: OmegaConf,
        tune_config: OmegaConf,
        tune_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        tune_logger: Optional[MultiProcessAdapter] = None,
    ):

        if tune_logger is None:
            tune_logger = get_logger(__name__, log_level="INFO")
        requires_eval = eval_loader is not None

        device = accelerator.device
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        elif accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16

        # ---- DataLoader preparation ---- #
        tune_batch_size = tune_loader.batch_size
        tune_loader = accelerator.prepare(tune_loader)
        if eval_loader is not None:
            eval_loader = accelerator.prepare(eval_loader)
        num_tune_samples = len(tune_loader) * tune_batch_size
        num_epochs = math.ceil(tune_config.max_tuning_steps / num_tune_samples)

        # ---- Pipeline, Adapter, and Optimizer preparation ---- #
        pipeline = Tuner.init_pipeline(pipe_config, weight_dtype, device)
        tune_modules = pipe_config.tune_modules

        optim_params = []
        accumulate_modules = []
        adapter_configs: AdapterConfigs = instantiate(adpt_config)

        for module in tune_modules:
            if module in pipeline.components:
                # Register AdapterManager for every linear layer in module
                register_adapter(pipeline.components[module])

                # Add sepcific adapter into module
                add_adapter(pipeline.components[module], adapter_configs, overwrite=False)

                # Enable gradients for training
                enable_adapter(pipeline.components[module], adapter_configs.adapter_name)
                optim_params.extend([p for p in pipeline.components[module].parameters() if p.requires_grad])

                # Active adapter by name (it will call adapter by name)
                activate_adapter(pipeline.components[module], adapter_configs.adapter_name)
                summary = Tuner.summarize_model(pipeline.components[module])
                
                for k, v in summary.items():
                    tune_logger.info(f"{k}: {v}")

                accumulate_modules.append(pipeline.components[module])
            else:
                tune_logger.warning(f"Module {module} cannot be found in pipeline components.")
        pipeline = pipeline.to(device, dtype=weight_dtype)

        optimizer = Prodigy(params=optim_params)
        optimizer, *accumulate_modules = accelerator.prepare(optimizer, *accumulate_modules)

        # ---- Record arguments ---- #
        global_steps = 0
        avg_step_loss = []
        step_bits = len(str(tune_config.max_tuning_steps))

        for epoch in range(num_epochs):
            for batch in tune_loader:
                # ---- Forward and Backward ---- #
                with accelerator.accumulate(*accumulate_modules):
                    with accelerator.autocast():
                        step_outputs: Dict[str, torch.Tensor] = tuning_step(
                            accelerator=accelerator,
                            pipeline=pipeline,
                            batch=batch,
                            generator=torch.Generator(device).manual_seed(tune_config.random_seed),
                            timestep_weighting_scheme=loss_config.timestep_weighting_scheme,
                            mu_logit_mean=loss_config.mu_logit_mean,
                            mu_logit_std=loss_config.mu_logit_std,
                            mu_mode_scale=loss_config.mu_mode_scale,
                            VAE_MAX_RESOLUTION=512 * 512,
                            CONDITION_MAX_RESOLUTION=384 * 384,
                            device=device,
                            weight_dtype=weight_dtype,
                        )

                    flow_matching_loss = step_outputs["flow_matching_loss"]
                    avg_step_loss.append(flow_matching_loss.cpu().item())
                    accelerator.backward(flow_matching_loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(optim_params, tune_config.max_grad_norm)

                    # TODO: Add learning rate scheduler if required
                    optimizer.step()
                    optimizer.zero_grad()
                # End accelerator accumulate

                # ---- For every synch steps, save or eval model ---- #
                if accelerator.sync_gradients:
                    global_steps += 1

                    avg_step_loss = sum(avg_step_loss) / len(avg_step_loss)
                    accelerator.log({"flow_matching_loss": avg_step_loss}, step=global_steps)

                    tune_logger.info(
                        f"Step [{global_steps:0{step_bits}d}/{tune_config.max_tuning_steps}] Loss {avg_step_loss:.6f}"
                    )
                    avg_step_loss = []

                    # ---- Save safetensors ---- #
                    if global_steps % tune_config.save_steps == 0:
                        os.makedirs(tune_config.save_dir, exist_ok=True)
                        for module in tune_modules:
                            save_name = f"{module}-{adapter_configs.adapter_name}.safetensors"
                            save_path = os.path.join(tune_config.save_dir, save_name)

                            state_dicts = {
                                n: p
                                for n, p in pipeline.components[module].named_parameters()
                                if adapter_configs.adapter_name in n
                            }
                            save_file(state_dicts, save_path)
                            tune_logger.info(
                                f"Adapter {adapter_configs.adapter_name} in Module {module} has been saved to {save_name}."
                            )

                    # ---- Evaluate model performance ---- #
                    if requires_eval and global_steps % tune_config.eval_steps == 0:
                        # TODO: Add evaluation steps
                        pass

                if global_steps >= tune_config.max_tuning_steps:
                    break
            # End one epoch
            if global_steps >= tune_config.max_tuning_steps:
                break
        # End tuning
        accelerator.wait_for_everyone()

        accelerator.end_training()
        tune_logger.info(f"✅ Tuning Finished~")
