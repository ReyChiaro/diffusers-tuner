import hydra
import torch
from hydra.utils import instantiate

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from diffusers_tuner.tuner import TuneConfigs, Tuner
from pipelines.pipeline_utils import PipelineConfigs, TunePipeline


@hydra.main(config_path="configs", config_name="test.yaml", version_base="v1.2")
def tune(cfgs: OmegaConf):

    dataset: Dataset = instantiate(cfgs.dataset)

    pipe_cfgs: PipelineConfigs = instantiate(cfgs.pipeline)
    pipeline = TunePipeline(pipe_cfgs, weight_dtype=torch.bfloat16, device="cuda")

    pipeline.add_adapter(
        cfgs.adapter,
        tune_modules=pipe_cfgs.tune_modules,
        requires_grad=True,
        adpt_checkpoint=None,
        weight_dtype=torch.bfloat16,
        device="cuda",
    )

    tuner_cfgs: TuneConfigs = instantiate(cfgs.tune)
    tuner = Tuner(tuner_cfgs)

    tuner.finetune(pipeline, cfgs.adapter.adapter_name, dataset, None)


if __name__ == "__main__":
    tune()
