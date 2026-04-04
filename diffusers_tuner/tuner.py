import os
import sys
import copy
import math
import yaml
import torch
import logging
import dataclasses
import torchvision.transforms.functional as T

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from hydra.utils import instantiate
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from prodigyopt import Prodigy
from torchvision.utils import save_image

from diffusers_tuner.tune_utils import (
    summarize_pipeline,
    find_accumulate_modules,
    find_trainable_params,
)
from pipelines.pipeline_utils import TunePipeline, ForwardOutputs


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

    random_seed: int
    max_tuning_steps: int
    mixed_precision: str
    grad_accumulate_steps: int
    max_grad_norm: float


class Tuner:

    def __init__(self, cfgss: TuneConfigs | OmegaConf):
        if not isinstance(cfgss, TuneConfigs):
            cfgss = instantiate(cfgss)
        self.cfgs = cfgss

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
    def evaluate_during_finetune(
        self,
        pipeline: TunePipeline,
        evalset: Dataset,
        device: torch.device,
        logger: logging.Logger,
        max_eval_num: int = 5,
    ):
        log_title = "-" * 21 + " Eval " + "-" * 22
        logger.info(
            f"\n{log_title}\nmax_eval_num: {max_eval_num}\nevaluate_dir: {self.cfgs.evaluation_dir}\n{'-' * len(log_title)}"
        )
        steps = 0
        infer_kwargs = {
            "inference_steps": 25,
            "return_dict": True,
            "output_type": "pil",
            "generator": torch.Generator(device).manual_seed(self.cfgs.random_seed),
        }
        for sample in evalset:
            steps += 1
            logger.info(f"Eval Step [{steps}/{max_eval_num}]")
            output: Image.Image = pipeline.handler.fn_auto_fill(
                pipeline.pipeline,
                sample,
            )(
                **infer_kwargs
            ).images[0]

            out_save_name = f"step-{steps}-output.jpg"
            con_save_name = f"step-{steps}-concat.jpg"

            # TODO Fix to use arbitrary keys
            conditions = [sample["image"]] if isinstance(sample["images"], torch.Tensor) else sample["images"]
            target = sample["target"]
            conditions = [T.resize(c, [target.height, target.width]) for c in conditions]

            output = T.to_tensor(output)
            save_image(output, os.path.join(self.cfgs.evaluation_dir, out_save_name))

            output = T.resize(output, [target.height, target.width])
            save_image(
                torch.cat(conditions + [output, target]),
                os.path.join(self.cfgs.evaluation_dir, con_save_name),
            )

            if steps >= max_eval_num:
                break

    def finetune(
        self,
        pipeline: TunePipeline,
        adapter_name: str,
        tune_loader: DataLoader,
        evalset: Dataset | None = None,
    ):
        cfgs = copy.deepcopy(self.cfgs)

        accelerator = Accelerator(
            split_batches=False,
            mixed_precision=cfgs.mixed_precision,
            gradient_accumulation_steps=cfgs.grad_accumulate_steps,
            log_with=cfgs.log_with,
            project_config=ProjectConfiguration(
                project_dir=cfgs.output_dir,
                logging_dir=cfgs.log_dir,
            ),
            kwargs_handlers=DistributedDataParallelKwargs(
                find_unused_parameters=True,
            ),
        )

        if accelerator.is_main_process:
            accelerator.init_trackers(cfgs.project_name)
        logger = self.setup_logger(
            is_main_process=accelerator.is_main_process,
            local_rank=accelerator.local_process_index,
        )

        log_title = "=" * 20 + " Configs " + "=" * 20
        logger.info(f"\n{log_title}\n{yaml.dump(dataclasses.asdict(cfgs))}\n{'=' * len(log_title)}")

        requires_eval = evalset is not None

        device = accelerator.device
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        elif accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16

        pipeline = pipeline.to(device=device, dtype=weight_dtype)

        # ---- DataLoader preparation ---- #
        tune_batch_size = tune_loader.batch_size
        tune_loader = accelerator.prepare(tune_loader)
        if evalset is not None:
            evalset = accelerator.prepare(evalset)
        num_tune_samples = len(tune_loader) * tune_batch_size
        num_epochs = math.ceil(cfgs.max_tuning_steps / num_tune_samples)

        trainable_params = find_trainable_params(pipeline)
        accumulate_modules = find_accumulate_modules(pipeline)

        summary = summarize_pipeline(pipeline)
        log_title = "-" * 20 + " Summary " + "-" * 20
        summary_str = "\n".join(f"{k}: {v}" for k, v in summary.items())
        logger.info(f"\n{log_title}\n{summary_str}\n{'-' * len(log_title)}")

        # ---- Optimizer and Scheduler (Optional) preparation ---- #
        optimizer = Prodigy(params=trainable_params)
        optimizer, *accumulate_modules = accelerator.prepare(optimizer, *accumulate_modules)

        # ---- Start Finetune ---- #
        global_steps = 0
        avg_step_loss = []
        step_bits = len(str(cfgs.max_tuning_steps))

        for epoch in range(num_epochs):
            for batch in tune_loader:
                # ---- Forward and Backward ---- #
                with accelerator.accumulate(*accumulate_modules):
                    with accelerator.autocast():
                        step_outputs: ForwardOutputs = pipeline(batch)
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

                    logger.info(f"Tune Step [{global_steps:0{step_bits}d}/{cfgs.max_tuning_steps}] Loss {avg_step_loss:.6f}")
                    avg_step_loss = []

                    # ---- Save safetensors ---- #
                    if global_steps % cfgs.save_steps == 0:
                        os.makedirs(cfgs.checkpoint_dir, exist_ok=True)
                        for module in pipeline.tune_modules:
                            save_name = f"{module}-{adapter_name}.safetensors"
                            save_path = os.path.join(cfgs.save_dir, save_name)

                            state_dicts = {
                                n: p for n, p in pipeline.components[module].named_parameters() if adapter_name in n
                            }
                            save_file(state_dicts, save_path)
                            logger.info(f"Adapter {adapter_name} in Module {module} has been saved to {save_name}.")

                    # ---- Evaluate model performance ---- #
                    if requires_eval and global_steps % cfgs.eval_steps == 0 and accelerator.is_main_process:
                        self.evaluate_during_finetune(pipeline, evalset)

                if global_steps >= cfgs.max_tuning_steps:
                    break
            if global_steps >= cfgs.max_tuning_steps:
                break
        # End tuning
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            self.evaluate_during_finetune(pipeline, evalset)

        accelerator.end_training()
        logger.info(f"✅ Tuning Finished~")
