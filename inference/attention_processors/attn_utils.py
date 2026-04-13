import torch

from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from typing import Callable

from pipelines.pipeline_utils import TunePipeline


def replace_processor(
    tune_pipeline: TunePipeline,
    module_name: str,
    attn_processor: AttentionProcessor | dict[str, AttentionProcessor],
    target_processors: list[str] | None = None,
):
    r"""
    :param module_name: The module where the attention processors should be replaced, view modules by `pipeline.components`.
    Example: transformer | unet
    :param target_processors: The module names that the processors will be replaced.
    Example: `transformer_blocks.0` will replace the processor of the first transformer block.
    """
    module: QwenImageTransformer2DModel = getattr(tune_pipeline.pipeline, module_name, None)
    assert module is not None, f"Module {module_name} is not loaded or not exist in current pipeline."
    assert hasattr(
        module, "set_attn_processor"
    ), f"Module {module_name} has no method named set_attn_processor. Failed replace processor."

    for proc_key, attn_proc in module.attn_processors.items():
        print(proc_key)



