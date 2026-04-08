import torch

from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from typing import Any, Optional

from pipelines.pipeline_utils import (
    ForwardHandler,
    ConditionOutputs,
    LatentOutputs,
    SampleOutputs,
    DenoiseOutputs,
    CriterionOutputs,
    ForwardOutputs,
)


class FluxFillForwardHandler(ForwardHandler):

    @staticmethod
    def check_inputs(pipeline: FluxFillPipeline, batch: dict[str, Any], **kwargs):
        images = batch.get("images", None)
        targets = batch.get("targets", None)
        prompts = batch.get("prompts", None)
        prompt_embeds = batch.get("prompt_embeds", None)
        pooled_prompt_embeds = batch.get("pooled_prompt_embeds", None)

        if images is None or targets is None:
            raise ValueError("`images` and `targets` are required for flux-fill training.")

        if prompts is None or prompts[0] is None:
            if prompt_embeds is None or pooled_prompt_embeds is None:
                raise ValueError("Both prompts and prompt_embeds/pooled_prompt_embeds are missing.")

    @staticmethod
    def encode_conditions(pipeline: FluxFillPipeline, batch: dict[str, Any], **kwargs) -> ConditionOutputs:
        device: torch.device = kwargs.get("device", pipeline.device)
        prompts: Optional[list[str]] = batch.get("prompts", None)
        prompt_2: Optional[list[str]] = batch.get("prompts_2", None)
        prompt_embeds: Optional[torch.Tensor] = batch.get("prompt_embeds", None)
        pooled_prompt_embeds: Optional[torch.Tensor] = batch.get("pooled_prompt_embeds", None)
        max_sequence_length: int = kwargs.get("max_sequence_length", 512)

        if prompts is None or prompts[0] is None:
            if prompt_embeds is None or pooled_prompt_embeds is None:
                raise ValueError("Both prompts and pre-computed prompt embeddings are missing.")

            if prompt_embeds.ndim == 2:
                prompt_embeds = prompt_embeds.unsqueeze(0)
            if pooled_prompt_embeds.ndim == 1:
                pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)

            dtype = pipeline.text_encoder.dtype if pipeline.text_encoder is not None else pipeline.transformer.dtype
            text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
            return ConditionOutputs(
                prompt_embeds=prompt_embeds.to(device=device),
                prompt_embeds_mask=None,
                others={
                    "pooled_prompt_embeds": pooled_prompt_embeds.to(device=device),
                    "text_ids": text_ids,
                },
            )

        encoded_prompt_embeds, encoded_pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=prompt_2,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
        return ConditionOutputs(
            prompt_embeds=encoded_prompt_embeds,
            prompt_embeds_mask=None,
            others={
                "pooled_prompt_embeds": encoded_pooled_prompt_embeds,
                "text_ids": text_ids,
            },
        )

    @staticmethod
    def encode_latents(pipeline: FluxFillPipeline, batch: dict[str, Any], **kwargs) -> LatentOutputs:
        device: torch.device = kwargs.get("device", pipeline.device)
        weight_dtype: torch.dtype = kwargs.get("weight_dtype", pipeline.transformer.dtype)
        generator: Optional[torch.Generator] = kwargs.get("generator", None)

        images: torch.Tensor = batch["images"].to(device=device, dtype=weight_dtype)
        targets: torch.Tensor = batch["targets"].to(device=device, dtype=weight_dtype)

        batch_size = targets.shape[0]
        if images.shape[0] % batch_size != 0:
            raise ValueError(
                f"`images` should be flattened as [num_cond_per_target * B, C, H, W]. "
                f"Got images.shape[0]={images.shape[0]}, B={batch_size}."
            )

        num_cond_per_target = images.shape[0] // batch_size
        if num_cond_per_target < 2:
            raise ValueError(
                "FluxFill requires image and mask conditions. "
                + f"At least 2 conditions are expected, got {num_cond_per_target}."
            )

        packed_conditions = images.view(batch_size, num_cond_per_target, *images.shape[1:])
        source_images = packed_conditions[:, 0]

        # NOTE: The highlighted part (mask=1) is to be editted.
        # And FluxFill will multiply image with (1-mask)
        # so we input 1-mask as the `mask` used in FluxFill
        mask_images = 1.0 - packed_conditions[:, 1]

        height = pipeline.vae_scale_factor * 2 * (int(targets.shape[-2]) // (pipeline.vae_scale_factor * 2))
        width = pipeline.vae_scale_factor * 2 * (int(targets.shape[-1]) // (pipeline.vae_scale_factor * 2))

        source_images = pipeline.image_processor.preprocess(source_images, height=height, width=width).to(
            device=device, dtype=weight_dtype
        )
        targets = pipeline.image_processor.preprocess(targets, height=height, width=width).to(
            device=device, dtype=weight_dtype
        )
        mask_images = pipeline.mask_processor.preprocess(mask_images, height=height, width=width).to(
            device=device, dtype=weight_dtype
        )

        target_latents = pipeline._encode_vae_image(image=targets, generator=generator)
        num_channels_latents = pipeline.vae.config.latent_channels
        target_latents = pipeline._pack_latents(
            target_latents,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height // pipeline.vae_scale_factor,
            width=width // pipeline.vae_scale_factor,
        )

        masked_images = source_images * (1 - mask_images)
        mask_latents, masked_image_latents = pipeline.prepare_mask_latents(
            mask=mask_images,
            masked_image=masked_images,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            num_images_per_prompt=1,
            height=height,
            width=width,
            dtype=weight_dtype,
            device=device,
            generator=generator,
        )
        masked_and_mask_latents = torch.cat((masked_image_latents, mask_latents), dim=-1)

        noise_latents = torch.randn_like(target_latents, device=device, dtype=weight_dtype)
        latent_image_ids = pipeline._prepare_latent_image_ids(
            batch_size=batch_size,
            height=height // pipeline.vae_scale_factor // 2,
            width=width // pipeline.vae_scale_factor // 2,
            device=device,
            dtype=weight_dtype,
        )

        return LatentOutputs(
            noise_latents=noise_latents,
            target_latents=target_latents,
            condition_latents=masked_and_mask_latents,
            others={
                "latent_image_ids": latent_image_ids,
            },
        )

    @staticmethod
    def sample_latents(pipeline: FluxFillPipeline, batch: dict[str, Any], **kwargs) -> SampleOutputs:
        target_latents: torch.Tensor = kwargs["target_latents"]
        noise_latents: torch.Tensor = kwargs["noise_latents"]
        generator: Optional[torch.Generator] = kwargs.get("generator", None)
        timestep_weighting_scheme: str = kwargs.get("timestep_weighting_scheme", "logit_normal")
        mu_logit_mean: float = kwargs.get("mu_logit_mean", 0.0)
        mu_logit_std: float = kwargs.get("mu_logit_std", 1.0)
        mu_mode_scale: float = kwargs.get("mu_mode_scale", 1.29)

        batch_size = target_latents.shape[0]
        device = target_latents.device
        weight_dtype = target_latents.dtype
        scheduler = pipeline.scheduler

        if scheduler.timesteps is None or len(scheduler.timesteps) == 0:
            scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)

        def _get_sigmas(timesteps: torch.Tensor, latent_ndim: int = 3) -> torch.Tensor:
            sched_sigmas = scheduler.sigmas.to(device=device)
            sched_timesteps = scheduler.timesteps.to(device=device)
            indices = [(sched_timesteps == t).nonzero().item() for t in timesteps]
            sigmas = sched_sigmas[indices].flatten()
            while sigmas.ndim < latent_ndim:
                sigmas = sigmas.unsqueeze(-1)
            return sigmas

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
        sigmas = _get_sigmas(timesteps, latent_ndim=target_latents.ndim).to(device=device, dtype=weight_dtype)
        noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise_latents
        return SampleOutputs(noisy_latents=noisy_latents, timesteps=timesteps, sigmas=sigmas)

    @staticmethod
    def denoise_forward(pipeline: FluxFillPipeline, batch: dict[str, Any], **kwargs) -> DenoiseOutputs:
        condition_outputs: ConditionOutputs = kwargs["condition_outputs"]
        latent_outputs: LatentOutputs = kwargs["latent_outputs"]
        sample_outputs: SampleOutputs = kwargs["sample_outputs"]
        joint_attention_kwargs: Optional[dict[str, Any]] = kwargs.get("joint_attention_kwargs", None)
        guidance_scale: float = kwargs.get("guidance_scale", 30.0)

        latents = sample_outputs.noisy_latents
        masked_and_mask_latents = latent_outputs.condition_latents
        hidden_states = torch.cat((latents, masked_and_mask_latents), dim=2)

        if pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([latents.shape[0]], guidance_scale, device=latents.device, dtype=torch.float32)
        else:
            guidance = None

        pred = pipeline.transformer(
            hidden_states=hidden_states,
            timestep=sample_outputs.timesteps / 1000,
            guidance=guidance,
            pooled_projections=condition_outputs.others["pooled_prompt_embeds"],
            encoder_hidden_states=condition_outputs.prompt_embeds,
            txt_ids=condition_outputs.others["text_ids"],
            img_ids=latent_outputs.others["latent_image_ids"],
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        return DenoiseOutputs(predictions=pred)

    @staticmethod
    def criterion_fn(pipeline: FluxFillPipeline, batch: dict[str, Any], **kwargs) -> CriterionOutputs:
        denoise_outputs: DenoiseOutputs = kwargs["denoise_outputs"]
        latent_outputs: LatentOutputs = kwargs["latent_outputs"]
        sample_outputs: SampleOutputs = kwargs["sample_outputs"]
        timestep_weighting_scheme: str = kwargs.get("timestep_weighting_scheme", "logit_normal")

        target_v = latent_outputs.noise_latents - latent_outputs.target_latents
        batch_size = target_v.shape[0]

        loss_w = compute_loss_weighting_for_sd3(
            weighting_scheme=timestep_weighting_scheme,
            sigmas=sample_outputs.sigmas,
        )
        flow_matching_loss = torch.mean(
            (loss_w.float() * (denoise_outputs.predictions.float() - target_v.float()) ** 2).reshape(batch_size, -1),
            dim=1,
        )
        return CriterionOutputs(loss=flow_matching_loss.mean())

    @staticmethod
    def forward_step(pipeline: FluxFillPipeline, batch: dict[str, Any], **kwargs) -> ForwardOutputs:
        device = pipeline.transformer.device
        weight_dtype = pipeline.transformer.dtype

        FluxFillForwardHandler.check_inputs(pipeline, batch)
        condition_outputs = FluxFillForwardHandler.encode_conditions(
            pipeline,
            batch,
            device=device,
        )
        latent_outputs = FluxFillForwardHandler.encode_latents(
            pipeline,
            batch,
            device=device,
            weight_dtype=weight_dtype,
        )
        sample_outputs = FluxFillForwardHandler.sample_latents(
            pipeline,
            batch,
            target_latents=latent_outputs.target_latents,
            noise_latents=latent_outputs.noise_latents,
        )
        denoise_outputs = FluxFillForwardHandler.denoise_forward(
            pipeline,
            batch,
            condition_outputs=condition_outputs,
            latent_outputs=latent_outputs,
            sample_outputs=sample_outputs,
        )
        criterion_outputs = FluxFillForwardHandler.criterion_fn(
            pipeline,
            batch,
            denoise_outputs=denoise_outputs,
            latent_outputs=latent_outputs,
            sample_outputs=sample_outputs,
        )
        return ForwardOutputs(
            loss=criterion_outputs.loss,
            predictions=denoise_outputs.predictions,
        )
