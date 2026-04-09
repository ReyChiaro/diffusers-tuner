import hydra
import torch

from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from pipelines.pipeline_utils import PipelineConfigs, TunePipeline


@hydra.main(config_path="configs", config_name="infer.yaml", version_base="v1.2")
def inference(cfgs: OmegaConf):

    pipe_cfgs: PipelineConfigs = instantiate(cfgs.pipeline)
    pipeline = TunePipeline(pipe_cfgs, weight_dtype=torch.bfloat16, device="cuda")

    pipeline.add_adapter(
        cfgs.adapter,
        tune_modules=pipe_cfgs.tune_modules,
        requires_grad=False,
        adpt_checkpoint=cfgs.adpt_checkpoint,
        weight_dtype=torch.bfloat16,
        device="cuda",
    )

    print(cfgs.adpt_checkpoint)
    print(cfgs.prompt)

    source_image = Image.open(cfgs.source_image).convert("RGB")
    mask_image = Image.open(cfgs.mask_image).convert("RGB")
    source_image.save("inference/source.jpg")
    mask_image.save("inference/mask.jpg")
    if hasattr(cfgs, "target_image"):
        target_image = Image.open(cfgs.target_image).convert("RGB")
        target_image.save("inference/target.jpg")

    output_image = pipeline.pipeline(
        prompt=cfgs.prompt,
        image=source_image,
        mask_image=mask_image,
        num_inference_steps=50,
        generator=torch.Generator("cuda").manual_seed(42),
        output_type="pil",
        return_dict=True,
    ).images[0]

    output_image.save("inference/output.jpg")


if __name__ == "__main__":
    inference()
