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
    pipeline = TunePipeline(pipe_cfgs, weight_dtype=torch.float32, device="cuda")

    tuner_cfgs: TuneConfigs = instantiate(cfgs.tune)
    tuner = Tuner(tuner_cfgs)
    tuner.prepare_prompt_embeds(pipeline, dataset, cfgs.prompt_embeds_save_dir)


if __name__ == "__main__":
    tune()
