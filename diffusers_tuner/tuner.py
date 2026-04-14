import os
import sys
import copy
import math
import yaml
import torch
import logging
import contextlib
import dataclasses
import torchvision.transforms.functional as T

from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, StateDictConfig, FullStateDictConfig
from hydra.utils import instantiate
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from loguru import logger
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from prodigyopt import Prodigy
from torchvision.utils import save_image
from typing import Any

from diffusers_tuner.tune_utils import (
    summarize_pipeline,
    find_accumulate_modules,
    find_trainable_params,
)
from pipelines.pipeline_utils import TunePipelineManager, ForwardOutputs, ConditionOutputs
from data.tune_dataset import BucketBatchSampler, DatasetSchema, DiffusersTunerDataset, DataConfigs


@dataclasses.dataclass
class TuneConfigs:

    project_name: str
    timestamp: str
    output_dir: str
    log_dir: str
    log_with: str
    log_per_rank: bool
    checkpoint_dir: str
    evaluation_dir: str
    save_steps: int
    eval_steps: int
    max_eval_num: int

    random_seed: int
    max_tuning_steps: int
    mixed_precision: str
    grad_accumulate_steps: int
    max_grad_norm: float

    data_cfgs: DataConfigs = dataclasses.field(default_factory=DataConfigs())


class Tuner:

    def __init__(self, cfgs: TuneConfigs | OmegaConf):
        if not isinstance(cfgs, TuneConfigs):
            cfgs = instantiate(cfgs)
        self.cfgs = cfgs

    def setup_logger(
        self,
        is_main_process: bool = True,
        local_rank: int = 0,
    ):
        logger.remove()

        log_name = f"{self.cfgs.timestamp}-rank{local_rank}.log"
        os.makedirs(self.cfgs.log_dir, exist_ok=True)
        file_path = os.path.join(self.cfgs.log_dir, log_name)

        if is_main_process:
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO",
                colorize=True,
            )
            logger.add(
                file_path,
                format="{time:YYYY-MM-DD HH:mm:ss} | [Rank {extra[rank]}] | {level: <8} | {name}:{line} - {message}",
                level="DEBUG",
                rotation="50 MB",
                enqueue=True,
            )

        elif self.cfgs.log_per_rank:
            logger.add(
                file_path,
                format="{time:YYYY-MM-DD HH:mm:ss} | [Rank {extra[rank]}] | {level: <8} | {name}:{line} - {message}",
                level="DEBUG",
                rotation="50 MB",
                enqueue=True,
            )

        return logger.bind(rank=local_rank)

    @torch.inference_mode()
    def prepare_prompt_embeds(
        self,
        accelerator: Accelerator,
        pipeline_manager: TunePipelineManager,
        dataset: DatasetSchema,
        prompt_embeds_save_dir: str,
        device: torch.device | None = None,
        weight_dtype: torch.dtype | None = None,
    ):
        device = device or accelerator.device

        if weight_dtype is None:
            weight_dtype = torch.float32
            if accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
            elif accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16

        cfgs = copy.deepcopy(self.cfgs)
        os.makedirs(prompt_embeds_save_dir, exist_ok=True)

        if accelerator.is_main_process:
            accelerator.init_trackers(cfgs.project_name)
        logger = self.setup_logger(
            is_main_process=accelerator.is_main_process,
            local_rank=accelerator.local_process_index,
        )

        log_title = "=" * 20 + " Configs " + "=" * 20
        logger.info(f"\n{log_title}\n{yaml.dump(dataclasses.asdict(cfgs))}\n{'=' * len(log_title)}")

        # ---- DataLoader preparation ---- #
        batch_size = self.cfgs.data_cfgs.batch_size
        num_workers = self.cfgs.data_cfgs.num_workers
        drop_last = self.cfgs.data_cfgs.drop_last
        data_loader = self.get_dataloader(
            dataset,
            world_size=accelerator.num_processes,
            rank=accelerator.process_index,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        data_loader = accelerator.prepare(data_loader)

        # ---- Pipeline preparation ---- #
        pipeline_manager.pipeline.to(device=device, dtype=weight_dtype)

        for i, batch in enumerate(data_loader):
            logger.info(f"Batch [{i+1}/{len(data_loader)}]")
            # ---- Forward and Backward ---- #
            conditions: ConditionOutputs = pipeline_manager.handler.encode_conditions(
                pipeline_manager.pipeline, batch, device=device
            )
            prompt_embeds: torch.Tensor = conditions.prompt_embeds
            prompt_embeds_mask: torch.Tensor = conditions.prompt_embeds_mask

            for b in range(batch_size):
                target_name = Path(batch["target_paths"][b]).stem

                states = {
                    "prompt_embeds": prompt_embeds[b],
                    "prompt_embeds_mask": prompt_embeds_mask,
                }
                save_file(states, os.path.join(prompt_embeds_save_dir, f"{target_name}.safetensors"))

    @torch.inference_mode()
    def evaluate_during_finetune(
        self,
        accelerator: Accelerator,
        pipeline_manager: TunePipelineManager,
        evalset: DatasetSchema,
        global_steps: int,
        device: torch.device,
        logger: logging.Logger,
        max_eval_num: int,
    ):
        log_title = "-" * 21 + " Eval " + "-" * 22
        evaluation_dir = os.path.join(self.cfgs.evaluation_dir, f"global-steps-{global_steps}")
        if accelerator.is_main_process:
            os.makedirs(evaluation_dir, exist_ok=True)

        logger.info(
            f"\n{log_title}\nmax_eval_num: {max_eval_num}\nevaluate_dir: {evaluation_dir}\n{'-' * len(log_title)}"
        )
        seed = self.cfgs.random_seed + accelerator.process_index
        infer_kwargs = {
            "inference_steps": 25,
            "return_dict": True,
            "output_type": "pil",
            "generator": torch.Generator(device).manual_seed(seed),
        }
        steps = 0

        modules_to_summon = [
            pipeline_manager.get_module(m)
            for m in pipeline_manager.adpt_tune_modules
            if isinstance(pipeline_manager.get_module(m), FSDP)
        ]

        with contextlib.ExitStack() as stack:
            for m in modules_to_summon:
                stack.enter_context(FSDP.summon_full_params(m))

            for i, sample in enumerate(evalset):
                if i % accelerator.num_processes != accelerator.process_index:
                    continue
                if i >= max_eval_num:
                    break

                steps += 1
                logger.info(f"Eval Step [{steps}/{max_eval_num}]")
                output: Image.Image = pipeline_manager.handler.fn_auto_fill(pipeline_manager.pipeline, sample)(
                    **infer_kwargs
                ).images[0]

                out_save_name = f"step-{steps}-rank{accelerator.process_index}-output.jpg"
                con_save_name = f"step-{steps}-rank{accelerator.process_index}-concat.jpg"

                # TODO Fix to use arbitrary keys
                conditions = [sample["image"]] if isinstance(sample["images"], torch.Tensor) else sample["images"]
                target = sample["target"]
                conditions = [T.resize(c, [target.height, target.width]) for c in conditions]

                output = T.to_tensor(output)
                save_image(output, os.path.join(evaluation_dir, out_save_name))

                output = T.resize(output, [target.height, target.width])
                save_image(
                    torch.cat(conditions + [output, target]),
                    os.path.join(evaluation_dir, con_save_name),
                )

    def get_dataloader(
        self,
        dataset: DatasetSchema,
        world_size: int | None = None,
        rank: int | None = None,
        batch_size: int = 1,
        num_workers: int = 4,
        drop_last: bool = False,
    ) -> DataLoader:
        batch_sampler = None
        collate_fn = dataset.collate_fn
        if getattr(dataset, "bucket_dataset", False):
            assert isinstance(
                dataset, DiffusersTunerDataset
            ), f"Bucket datset is enabled, dataset should be TuneBucketDataset."
            assert (
                world_size is not None and rank is not None
            ), f"Bucket dataset needs pass world size and local rank as argments."
            batch_sampler = BucketBatchSampler(
                bucket_indices=dataset.buckets,
                batch_size=batch_size,
                num_replicas=world_size,
                rank=rank,
                drop_last=drop_last,
                seed=self.cfgs.random_seed,
            )
            dataloader = DataLoader(
                dataset=dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                prefetch_factor=1,
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )
        return dataloader

    def save_state_dict(
        self,
        accelerator: Accelerator,
        pipeline_manager: TunePipelineManager,
        global_steps: int,
        step_bits: int | None = None,
    ) -> str:
        step_bits = step_bits or len(str(global_steps)) + 1
        checkpoint_path = os.path.join(self.cfgs.checkpoint_dir, f"global-steps-{global_steps}")
        if accelerator.is_main_process:
            os.makedirs(checkpoint_path, exist_ok=True)

        if pipeline_manager.adpt_tune_modules:
            # Adapter based tuning, adpt_tune_modules is the list of component names
            for module_name in pipeline_manager.adpt_tune_modules:
                module = pipeline_manager.get_module(module_name)

                if accelerator.distributed_type == DistributedType.FSDP:
                    logger.debug(f"FSDP is enabled, save full adapters on rank 0.")
                    save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)

                    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, save_policy):
                        state_dicts = module.state_dict()
                else:
                    unwrapped_module = accelerator.unwrap_model(module)
                    state_dicts = unwrapped_module.state_dict()

                if accelerator.is_main_process:
                    adapter_name = pipeline_manager.adpt_configs.adapter_name
                    adapter_state_dicts = {n: p for n, p in state_dicts.items() if adapter_name in n}
                    if len(adapter_state_dicts) > 0:
                        save_name = f"{module_name}-{adapter_name}-step{global_steps:0{step_bits}d}.safetensors"
                        save_path = os.path.join(self.cfgs.checkpoint_dir, save_name)
                        save_file(adapter_state_dicts, save_path)
                        logger.info(f"Adapter {adapter_name} in Module {module_name} has been saved to {save_name}.")
                    else:
                        logger.warning(f"No parameters to save found in adapter {adapter_name}.")

        if pipeline_manager.full_tune_modules:
            # Tuning original parameters, full_tune_modules is the dict mapping
            # component name to module_names (e.g. {transformer: [transformer_blocks.59.attn]})
            # For this condition, the modules will be saved by shards on current device.
            shard_bit = len(str(accelerator.num_processes))
            for module_name in pipeline_manager.full_tune_modules:
                module = pipeline_manager.get_module(module_name)

                rank = accelerator.process_index
                if accelerator.distributed_type == DistributedType.FSDP:
                    logger.debug(f"FSDP is enabled, save model shards on {rank}.")

                    with FSDP.state_dict_type(module, StateDictType.SHARDED_STATE_DICT):
                        sharded_state_dicts = module.state_dict()
                    
                    shard_name = f"{module_name}-step{global_steps:0{step_bits}d}-shard{accelerator.process_index:0{shard_bit}d}.safetensors"
                    save_path = os.path.join(checkpoint_path, shard_name)

                    save_file(sharded_state_dicts, shard_name)
                    logger.info(f"Module {module_name} has been saved to {shard_name}.")
                else:
                    if accelerator.is_main_process:
                        unwrapped_module = accelerator.unwrap_model(module)
                        full_osd = unwrapped_module.state_dict()
                        save_name = f"{module_name}-step{global_steps:0{step_bits}d}-full.safetensors"
                        save_path = os.path.join(self.cfgs.checkpoint_dir, save_name)
                        save_file(full_osd, save_path)
                        logger.info(f"Module {module_name} has been saved to {save_name}.")

        accelerator.wait_for_everyone()

    def finetune(
        self,
        accelerator: Accelerator,
        pipeline_manager: TunePipelineManager,
        tuneset: DatasetSchema,
        evalset: DatasetSchema | None = None,
        device: torch.device | None = None,
        weight_dtype: torch.dtype | None = None,
    ):
        device = device or accelerator.device

        if weight_dtype is None:
            weight_dtype = torch.float32
            if accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
            elif accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16

        cfgs = copy.deepcopy(self.cfgs)

        if accelerator.is_main_process:
            accelerator.init_trackers(cfgs.project_name)
        logger = self.setup_logger(
            is_main_process=accelerator.is_main_process,
            local_rank=accelerator.process_index,
        )

        requires_eval = evalset is not None

        # ---- DataLoader preparation ---- #
        batch_size = cfgs.data_cfgs.batch_size
        num_workers = cfgs.data_cfgs.num_workers
        drop_last = cfgs.data_cfgs.drop_last
        tune_loader = self.get_dataloader(
            tuneset,
            world_size=accelerator.num_processes,
            rank=accelerator.process_index,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
        )

        # Accelerator may change the length of loader, we should re-calculate steps
        tune_loader = accelerator.prepare(tune_loader)
        num_tune_samples = len(tune_loader) * batch_size
        max_tuning_steps = math.ceil(cfgs.max_tuning_steps / batch_size)
        save_steps = math.ceil(cfgs.save_steps / batch_size)
        eval_steps = math.ceil(cfgs.eval_steps / batch_size)

        max_tuning_steps_per_process = math.ceil(max_tuning_steps / accelerator.num_processes)
        save_steps_per_process = math.ceil(save_steps / accelerator.num_processes)
        eval_steps_per_process = math.ceil(eval_steps / accelerator.num_processes)
        num_epochs = math.ceil(max_tuning_steps * batch_size / num_tune_samples)

        step_info = (
            f"{num_tune_samples=}\n{batch_size=}\n{max_tuning_steps=}\n{save_steps=}\n{eval_steps=}\n{num_epochs=}"
        )
        step_info += f"\n{max_tuning_steps_per_process=}\n{save_steps_per_process=}\n{eval_steps_per_process=}"

        # ---- Pipeline preparation ---- #
        pipeline_manager.pipeline.to(device=device, dtype=weight_dtype)
        trainable_params = find_trainable_params(pipeline_manager.pipeline)
        accumulate_modules = find_accumulate_modules(pipeline_manager.pipeline)

        # For FSDP mode, we must tell which sub-modules do not need gradient
        if accelerator.distributed_type == DistributedType.FSDP:
            ignored_modules = []
            for module in accumulate_modules:
                for m in module.modules():
                    if len(list(m.parameters())) > 0 and not any(p.requires_grad for p in m.parameters()):
                        ignored_modules.append(m)
            accelerator.state.fsdp_plugin.ignored_modules = ignored_modules

        summary = summarize_pipeline(pipeline_manager.pipeline)
        log_title = "-" * 20 + " Summary " + "-" * 20
        log_split = list("-" * len(log_title))
        log_split[1::2] = ["~"] * len(log_split[1::2])
        log_split = "".join(log_split)
        summary_info = "\n".join(f"{k}: {v}" for k, v in summary.items())
        logger.info(f"\n{log_title}\n{summary_info}\n{log_split}\n{step_info}\n{'-' * len(log_title)}")

        # ---- Optimizer and Scheduler (Optional) preparation ---- #
        optimizer = Prodigy(params=trainable_params)
        optimizer, *accumulate_modules = accelerator.prepare(optimizer, *accumulate_modules)

        # ---- Start Finetune ---- #
        global_steps = 0
        avg_step_loss = []
        step_bits = len(str(max_tuning_steps))

        for epoch in range(num_epochs):
            for batch in tune_loader:
                # ---- Forward and Backward ---- #
                with accelerator.accumulate(*accumulate_modules):
                    with accelerator.autocast():
                        step_outputs: ForwardOutputs = pipeline_manager(batch)
                    loss = step_outputs.loss
                    avg_step_loss.append(loss.cpu().item())
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, cfgs.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()
                # End accelerator accumulate

                # ---- For every synch steps, save or eval model ---- #
                if accelerator.sync_gradients:
                    global_steps += 1
                    avg_step_loss = sum(avg_step_loss) / len(avg_step_loss)

                    accelerator.log({"loss": avg_step_loss}, step=global_steps)
                    logger.info(f"Tune Step [{global_steps:0{step_bits}d}/{max_tuning_steps}] Loss {avg_step_loss:.6f}")
                    avg_step_loss = []

                    # ---- Save safetensors ---- #
                    if global_steps % save_steps == 0:
                        self.save_state_dict(
                            accelerator=accelerator,
                            pipeline_manager=pipeline_manager,
                            global_steps=global_steps,
                            step_bits=step_bits,
                        )

                    # ---- Evaluate model performance ---- #
                    if requires_eval and global_steps % eval_steps == 0:
                        self.evaluate_during_finetune(
                            accelerator=accelerator,
                            pipeline_manager=pipeline_manager,
                            evalset=evalset,
                            global_steps=global_steps,
                            device=device,
                            logger=logger,
                            max_eval_num=self.cfgs.max_eval_num,
                        )

                if global_steps >= max_tuning_steps:
                    break
            if global_steps >= max_tuning_steps:
                break
        # End tuning
        accelerator.wait_for_everyone()

        if requires_eval:
            self.evaluate_during_finetune(
                accelerator=accelerator,
                pipeline_manager=pipeline_manager,
                evalset=evalset,
                global_steps=global_steps,
                device=device,
                logger=logger,
                max_eval_num=self.cfgs.max_eval_num,
            )
        accelerator.end_training()
        logger.info(f"✅ Tuning Finished~")
