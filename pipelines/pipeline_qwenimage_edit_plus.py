import torch

from accelerate import Accelerator
from copy import deepcopy
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    QwenImageEditPlusPipeline,
    calculate_dimensions,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

from omegaconf import OmegaConf
from typing import List, Dict, Any, Optional


def init_pipeline(
    pipe_config: OmegaConf,
    weight_dtype: torch.dtype,
    device: torch.device,
) -> QwenImageEditPlusPipeline:
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pipe_config.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        pipe_config.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    ).to(device)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        pipe_config.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in transformer.parameters():
        p.requires_grad_(False)
    processor = tokenizer = text_encoder = None
    if pipe_config.load_text_encoder:
        processor = Qwen2VLProcessor.from_pretrained(
            pipe_config.pretrained_model_name_or_path,
            subfolder="processor",
        )
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pipe_config.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pipe_config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
        ).to(device)
        for p in text_encoder.parameters():
            p.requires_grad_(False)
    pipeline = QwenImageEditPlusPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer
    )

    return pipeline


def tuning_step(
    accelerator: Accelerator,
    pipeline: QwenImageEditPlusPipeline,
    batch: Dict[str, Any],
    generator: Optional[torch.Generator] = None,
    timestep_weighting_scheme: str = "logit_normal",
    mu_logit_mean: float = 0.0,
    mu_logit_std: float = 0.02,
    mu_mode_scale: float = 1.29,
    VAE_MAX_RESOLUTION: int = VAE_IMAGE_SIZE,
    CONDITION_MAX_RESOLUTION: int = CONDITION_IMAGE_SIZE,
    device: Optional[torch.device] = None,
    weight_dtype: Optional[torch.dtype] = None,
):
    r"""
    Args:
        :param pipeline (DiffusersPipeline or subclass): If tuning is required, then the tuning modules in this pipeline have been initialized.
        :param batch (Dict[str, Any]): We recommend the batch includes:
            - images (List[torch.Tensor], Required)
            - prompts (str, Optional)
            - targets (torch.Tensor, Required): For current model, single image output is better.
            - negative_prompts (str, Optional)
            If no prompts but prepared embeddings are provided:
            - prompt_embeds (torch.Tensor, Optional)
            - prompt_embeds_mask (torch.Tensor, Optional)
            - negative_prompt_embeds (torch.Tensor, Optional)
            - negative_prompt_embeds_mask (torch.Tensor, Optional)
        :param VAE_MAX_RESOLUTION (int): For QwenImage, a max resolution is recommended to reshape the image into a suitable same resolution with original one while not exceeds the max resolution.
        :param CONDITION_MAX_RESOLUTION (int): For QwenImage, the language encoder is an VLM, which will take both prompts and images as input, so the images should be reshaped to a smaller resolution.

    """
    # ---- Prepare useful arguments ---- #
    device = device or accelerator.device
    if weight_dtype is None:
        if accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        elif accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32

    images: List[torch.Tensor] = batch.get("images", None)
    targets: torch.Tensor = batch.get("targets", None)
    if images is None or targets is None:
        raise ValueError("images or targets is None.")
    prompts: List[str] = batch.get("prompts", None)
    prompt_embeds: torch.Tensor = batch.get("prompt_latents", None)
    prompt_embeds_mask: torch.Tensor = batch.get("prompt_latents_mask", None)

    batch_size = len(targets)
    num_image_per_prompt = len(images)
    multiple_of = pipeline.vae_scale_factor * 2
    target_h = targets[0].shape[-2]
    target_w = targets[0].shape[-1]
    target_h = target_h // multiple_of * multiple_of
    target_w = target_w // multiple_of * multiple_of
    scheduler = deepcopy(pipeline.scheduler)

    # ---- Prepare prompt embeds ---- #
    if prompts[0] is None:
        if prompt_embeds is None or prompt_embeds_mask is None:
            raise ValueError("Both prompts and prompt_embeds are None.")
    else:
        # prompts is not None, whether the embeds/mask is given, we encode the prompts.
        # Reshape images into condition shapes
        # TODO: Udpate to resolution-reserve encoding
        condition_w, condition_h = calculate_dimensions(CONDITION_MAX_RESOLUTION, images[0].shape[-1] / images[0].shape[-2])
        # condition_images is List of 4-D tensors (B, C, H, W)
        condition_images = [
            pipeline.image_processor.preprocess(
                image,
                height=condition_h,
                width=condition_w,
            )
            for image in images
        ]

        prompt_embeds = []
        prompt_embeds_mask = []

        for b in range(batch_size):
            embed, embed_mask = pipeline.encode_prompt(
                prompt=prompts[b],
                image=[condition_images[i][b] for i in range(num_image_per_prompt)],
                device=device,
                num_images_per_prompt=1,
                prompt_embeds=None,
                prompt_embeds_mask=None,
            )
            prompt_embeds.append(embed)
            prompt_embeds_mask.append(embed_mask)

        prompt_embeds = torch.stack(prompt_embeds)
        prompt_embeds_mask = torch.stack(prompt_embeds_mask)

    # ---- Prepare image input latents ---- #
    vae_images = []
    vae_shapes = []
    num_channels = pipeline.transformer.config.in_channels // 4
    for image in images:
        vae_w, vae_h = calculate_dimensions(VAE_MAX_RESOLUTION, image.shape[-1] / image.shape[-2])
        vae_image = pipeline.image_processor.preprocess(
            image=image,
            height=vae_h,
            width=vae_w,
        ).unsqueeze(2)
        vae_images.append(vae_image)
        vae_shapes.append((vae_h, vae_w))
    targets = targets.unsqueeze(2)

    eps_latents = []
    img_latents = []
    for b in range(batch_size):
        n_latent, i_laetent = pipeline.prepare_latents(
            images=[targets[b].unsqueeze(0)] + [vae_images[i][b].unsqueeze(0) for i in range(num_image_per_prompt)],
            num_channels_latents=num_channels,
            height=target_h,
            width=target_w,
            dtype=weight_dtype,
            device=device,
            generator=generator,
            batch_size=batch_size,
        )
        eps_latents.append(n_latent)
        img_latents.append(i_laetent)

    eps_latents = torch.cat(eps_latents)
    img_latents = torch.cat(img_latents)

    target_latents = img_latents[:, :(target_h // multiple_of) * (target_w // multiple_of)]
    img_latents = img_latents[:, (target_h // multiple_of) * (target_w // multiple_of) :]

    image_shapes = [
        [
            (1, target_h // multiple_of, target_w // multiple_of),
            *[(1, vae_h // multiple_of, vae_w // multiple_of) for vae_h, vae_w in vae_shapes],
        ]
    ] * batch_size

    # ---- Prepare timesteps ---- #
    def _get_sigmas(timesteps: torch.Tensor, latent_ndim: int = 4) -> torch.Tensor:
        device = timesteps.device
        sched_sigmas = scheduler.sigmas.to(device)
        sched_timesteps = scheduler.timesteps.to(device)
        indices = [(sched_timesteps == t).nonzero().item() for t in timesteps]
        sigmas = sched_sigmas[indices].flatten()
        while sigmas.ndim < latent_ndim:
            sigmas = sigmas.unsqueeze(-1)
        return sigmas

    # Sample timesteps (integer in [0, 1000]) and sigmas (float in [0, 1])
    u = compute_density_for_timestep_sampling(
        weighting_scheme=timestep_weighting_scheme,
        batch_size=batch_size,
        logit_mean=mu_logit_mean,
        logit_std=mu_logit_std,
        mode_scale=mu_mode_scale,
        device=device,
        generator=generator,
    )
    indices = (u * scheduler.config.num_train_timesteps).long().to("cpu")
    timesteps = scheduler.timesteps[indices].to(device)
    sigmas = _get_sigmas(timesteps, latent_ndim=target_latents.ndim).to(device, dtype=weight_dtype)

    # Shape (B, L, C)
    noisy_latents = (1.0 - sigmas) * target_latents + sigmas * eps_latents
    prompt_seq_lengths = prompt_embeds_mask.sum(dim=1).tolist()

    # Predict vector field
    latents = torch.cat([noisy_latents, img_latents], dim=-2)
    pred_v = pipeline.transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        timestep=timesteps / 1000,
        img_shapes=image_shapes,
        txt_seq_lens=prompt_seq_lengths,
        attention_kwargs={},
        return_dict=False,
    )[0]
    pred_v = pred_v[:, : noisy_latents.shape[1]]

    # pred_v = pipeline._unpack_latents(
    #     pred_v,
    #     height=target_h,
    #     width=target_w,
    #     vae_scale_factor=pipeline.vae_scale_factor,
    # )
    target_v = eps_latents - target_latents

    loss_w = compute_loss_weighting_for_sd3(
        weighting_scheme=timestep_weighting_scheme,
        sigmas=sigmas,
    )

    flow_matching_loss = torch.mean(
        (loss_w.float() * (pred_v.float() - target_v.float()) ** 2).reshape(batch_size, -1),
        dim=1,
    )
    flow_matching_loss = flow_matching_loss.mean()

    return {
        "flow_matching_loss": flow_matching_loss,
        "predict_v": pred_v,
    }
