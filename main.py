import hydra

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_module import DataModule
from tuner import Tuner


@hydra.main(config_path="configs", config_name="main", version_base="v1.2")
def finetune(cfgs: OmegaConf) -> None:
    # ---- Accelerator ---- #
    accelerator = Accelerator(
        mixed_precision=cfgs.accelerator.mixed_precision,
        gradient_accumulation_steps=cfgs.accelerator.gradient_accumulation_steps,
        log_with=cfgs.accelerator.log_with,
        project_config=ProjectConfiguration(cfgs.output_dir, logging_dir=cfgs.logging_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    is_main_process = accelerator.is_main_process

    logger = get_logger(__name__, log_level="INFO")

    log_title = "=" * 25 + " Configs " + "=" * 25
    logger.info(f"\n{log_title}\n{OmegaConf.to_yaml(cfgs)}" + "=" * len(log_title))

    if is_main_process:
        accelerator.init_trackers(project_name=cfgs.project_name)
    
    # ---- Data and Tuner ---- #
    data_module: DataModule = instantiate(cfgs.data_module)    
    tuner: Tuner = instantiate(cfgs.tuner)

    # ---- Start Finetune ---- #
    tuner.finetune(
        accelerator=accelerator,
        tune_loader=data_module.tune_loader,
        eval_loader=data_module.eval_loader,
        pipe_config=cfgs.pipeline,
        tune_logger=logger,
    )

    logger.info("✅ Tuner task finished.")


if __name__ == "__main__":
    finetune()
