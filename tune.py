import hydra
import torch

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, FullyShardedDataParallelPlugin
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from loguru import logger

from diffusers_tuner.tuner import TuneConfigs, Tuner
from pipelines.pipeline_utils import PipelineConfigs, TunePipelineManager


@hydra.main(config_path="configs", config_name="tune", version_base="v1.2")
def tune(cfgs: OmegaConf):

    log_title = "=" * 20 + " Configs " + "=" * 20
    logger.info(f"\n{log_title}\n{OmegaConf.to_yaml(cfgs)}\n" + "=" * len(log_title))

    accelerator = Accelerator(
        split_batches=False,
        mixed_precision=cfgs.tune.mixed_precision,
        gradient_accumulation_steps=cfgs.tune.grad_accumulate_steps,
        log_with=cfgs.tune.log_with,
        project_config=ProjectConfiguration(
            project_dir=cfgs.tune.output_dir,
            logging_dir=cfgs.tune.log_dir,
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        fsdp_plugin=FullyShardedDataParallelPlugin(ignored_modules=[]),
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    dataset: Dataset = instantiate(cfgs.dataset)
    pipe_cfgs: PipelineConfigs = instantiate(cfgs.pipeline)
    pipeline = TunePipelineManager(
        pipe_cfgs,
        weight_dtype=weight_dtype,
        device=accelerator.device,
    )

    if getattr(cfgs, "adapter", None) is not None:
        # Adapter tuning
        pipeline.add_adapter(
            cfgs.adapter,
            tune_modules=pipe_cfgs.adpt_tune_modules,
            requires_grad=True,
            adpt_checkpoint=None,
            weight_dtype=weight_dtype,
            device=accelerator.device,
        )
    else:
        # Tune module parameters
        pipeline.enable_full_tune_modules(
            cfgs.pipeline.full_tune_modules,
            weight_dtype=weight_dtype,
            device=accelerator.device,
        )

    tuner_cfgs: TuneConfigs = instantiate(cfgs.tune)
    tuner = Tuner(tuner_cfgs)

    tuner.finetune(
        accelerator=accelerator,
        pipeline_manager=pipeline,
        adapter_name=cfgs.adapter.adapter_name,
        tuneset=dataset,
        evalset=dataset,  # Use trainset to evaluate during training
        device=accelerator.device,
        weight_dtype=weight_dtype,
    )


if __name__ == "__main__":
    tune()
