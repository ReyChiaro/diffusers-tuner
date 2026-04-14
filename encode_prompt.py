import hydra
import torch

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from diffusers_tuner.tuner import TuneConfigs, Tuner
from pipelines.pipeline_utils import PipelineConfigs, TunePipeline


@hydra.main(config_path="configs", config_name="infer", version_base="v1.2")
def tune(cfgs: OmegaConf):

    accelerator = Accelerator()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    dataset: Dataset = instantiate(cfgs.dataset)

    pipe_cfgs: PipelineConfigs = instantiate(cfgs.pipeline)
    pipeline = TunePipeline(
        pipe_cfgs,
        weight_dtype=weight_dtype,
        device=accelerator.device,
    )

    tuner_cfgs: TuneConfigs = instantiate(cfgs.tune)
    tuner = Tuner(tuner_cfgs)
    tuner.prepare_prompt_embeds(
        accelerator,
        pipeline,
        dataset,
        cfgs.prompt_embeds_save_dir,
        device=accelerator.device,
        weight_dtype=weight_dtype,
    )


if __name__ == "__main__":
    tune()
