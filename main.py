import hydra

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from accelerate.logging import get_logger

from omegaconf import OmegaConf


@hydra.main(config_path="configs", config_name="main", version_base="v1.2")
def finetune(cfgs: OmegaConf) -> None:
    accelerator = Accelerator(
        mixed_precision=cfgs.accelerator.mixed_precision,
        gradient_accumulation_steps=cfgs.accelerator.gradient_accumulation_steps,
        log_with=cfgs.accelerator.log_with,
        project_config=ProjectConfiguration(cfgs.output_dir, logging_dir=cfgs.logging_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    logger = get_logger(__name__, log_level="INFO")

    log_title = "=" * 25 + " Configs " + "=" * 25
    logger.info(f"\n{log_title}\n{OmegaConf.to_yaml(cfgs)}" + "=" * len(log_title))


if __name__ == "__main__":
    finetune()
