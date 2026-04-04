import torch
import torch.nn as nn

from diffusers.pipelines.pipeline_utils import DiffusionPipeline


def freeze_parameters(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def freeze_module(pipeline: DiffusionPipeline, module_name: str):
    if isinstance(pipeline.components[module_name], nn.Module):
        freeze_parameters(pipeline.components[module_name])


def find_trainable_params(pipeline: DiffusionPipeline) -> list[torch.Tensor]:
    trainable_params = []
    for module in pipeline.components:
        if not isinstance(pipeline.components[module], nn.Module):
            continue
        for p in pipeline.components[module].parameters():
            if p.requires_grad:
                trainable_params.append(p)
    return trainable_params


def find_accumulate_modules(pipeline: DiffusionPipeline) -> list[nn.Module]:
    accum_modules = []
    for module in pipeline.components:
        if not isinstance(pipeline.components[module], nn.Module):
            continue
        for p in pipeline.components[module].parameters():
            # Add module to gradient accumulation as long as
            # there at least one parameter should to be trained
            if p.requires_grad:
                accum_modules.append(pipeline.components[module])
                break
    return accum_modules


def summarize_model(model: nn.Module) -> dict[str, float]:
    # Calculate parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate size on disk/memory (assuming float32 = 4 bytes)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio(%)": trainable_params / total_params,
        "model_size_mb(MB)": size_all_mb,
    }


def summarize_pipeline(pipeline: DiffusionPipeline) -> dict[str, float]:
    module_summarizes = []
    for module in pipeline.components:
        if not isinstance(pipeline.components[module], nn.Module):
            continue
        module_summarizes.append(summarize_model(pipeline.components[module]))
    total_params = sum([m["total_params"] for m in module_summarizes])
    trainable_params = sum([m["trainable_params"] for m in module_summarizes])
    model_size_mb = sum([m["model_size_mb(MB)"] for m in module_summarizes])
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio(%)": trainable_params / total_params,
        "model_size_mb(MB)": model_size_mb,
    }
