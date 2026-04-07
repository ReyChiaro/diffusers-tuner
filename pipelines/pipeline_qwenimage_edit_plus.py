import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
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
from typing import Any
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


class QwenImageEditPlusForwardHandler(ForwardHandler):

    @staticmethod
    def check_inputs(pipeline: QwenImageEditPlusPipeline, batch: dict[str, Any], **kwargs):
        images = batch.get("images", None)
        targets = batch.get("targets", None)
        prompts = batch.get("prompts", None)
        prompt_embeds = batch.get("prompt_embeds", None)
        prompt_embeds_mask = batch.get("prompt_embeds_mask", None)
        if images is None or targets is None:
            raise ValueError("`images` and `targets` are required for qwen-image-edit training.")
        # if not isinstance(images, list) or len(images) == 0:
        #     raise ValueError("`images` should be a non-empty list of input image tensors.")
        # Either raw prompts or pre-computed prompt latents are required.
        if prompts is None or prompts[0] is None:
            if prompt_embeds is None or prompt_embeds_mask is None:
                raise ValueError("Both prompts and prompt_embeds/prompt_embeds_mask are missing.")

        return ForwardHandler.fn_auto_fill(pipeline.check_inputs, batch, **kwargs)

    @staticmethod
    def encode_conditions(pipeline: QwenImageEditPlusPipeline, batch: dict[str, Any], **kwargs) -> ConditionOutputs:
        device: torch.device = kwargs.get("device", pipeline.device)
        prompts: list[str] = batch.get("prompts", None)
        images: torch.Tensor = batch.get("images", None)
        prompt_embeds: Optional[torch.Tensor] = batch.get("prompt_embeds", None)
        prompt_embeds_mask: Optional[torch.Tensor] = batch.get("prompt_embeds_mask", None)

        if prompts is None or prompts[0] is None:
            if prompt_embeds is None or prompt_embeds_mask is None:
                raise ValueError("Both prompts and pre-computed prompt latents are missing.")
            if prompt_embeds.ndim == 2:
                prompt_embeds = prompt_embeds.unsqueeze(0)
            if prompt_embeds_mask.ndim == 3:
                prompt_embeds_mask = prompt_embeds_mask.squeeze(0)
            return ConditionOutputs(
                prompt_embeds=prompt_embeds.to(device=device),
                prompt_embeds_mask=prompt_embeds_mask.to(device=device),
            )

        condition_w, condition_h = calculate_dimensions(
            CONDITION_IMAGE_SIZE,
            images.shape[-1] / images.shape[-2],
        )
        condition_images = pipeline.image_processor.resize(
            images,
            height=condition_h,
            width=condition_w,
        )

        batch_size = len(prompts)
        weight_dtype = images.dtype

        # We suppose all samples share the same `num_image_per_prompt`
        num_cond_per_prompt = condition_images.shape[0] // batch_size
        prompt_embeds = []
        prompt_embeds_mask = []

        # Qwen-Image-Edit-Plus encode_prompt do not support batch encoding
        # See https://github.com/huggingface/diffusers/blob/8070f6ec54a7699d5ee285090d9735d9c9b205d7/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L243
        # this line means that the image prompt templates are concatenated
        # based on the num of conditions used for ONE prompt.
        # So we re-implement this to support multiple images input.

        # 1. Prepare prompt and insert image placeholder into prompt template
        prompt_template = pipeline.prompt_template_encode
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        for b in range(batch_size):
            img_prompt = ""
            for i in range(num_cond_per_prompt):
                img_prompt += img_prompt_template.format(i + 1)
            prompts[b] = prompt_template.format(img_prompt + prompts[b])

        # 2. Tokenize
        model_inputs = pipeline.processor(
            text=prompts,
            images=condition_images,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = pipeline.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        drop_idx = pipeline.prompt_template_encode_start_idx
        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = pipeline._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        prompt_embeds_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])

        prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device)

        return ConditionOutputs(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )

    @staticmethod
    def encode_latents(pipeline: QwenImageEditPlusPipeline, batch: dict[str, Any], **kwargs) -> LatentOutputs:
        device: torch.device = kwargs.get("device", pipeline.device)
        weight_dtype: torch.dtype = kwargs.get("weight_dtype", pipeline.transformer.dtype)
        generator: Optional[torch.Generator] = kwargs.get("generator", None)
        vae_max_resolution: int = kwargs.get("vae_max_resolution", VAE_IMAGE_SIZE)
        images: torch.Tensor = batch["images"]
        targets: torch.Tensor = batch["targets"].to(device=device, dtype=weight_dtype)

        batch_size = targets.shape[0]
        num_cond_per_target = images.shape[0] // batch_size
        multiple_of = pipeline.vae_scale_factor * 2
        target_h = targets[0].shape[-2] // multiple_of * multiple_of
        target_w = targets[0].shape[-1] // multiple_of * multiple_of

        vae_target_w, vae_target_h = calculate_dimensions(vae_max_resolution, target_w / target_h)
        vae_targets = pipeline.image_processor.preprocess(
            image=targets,
            height=vae_target_h,
            width=vae_target_w,
        ).unsqueeze(
            2
        )  # (B,C,F,H,W)

        vae_w, vae_h = calculate_dimensions(vae_max_resolution, images.shape[-1] / images.shape[-2])
        vae_images = pipeline.image_processor.preprocess(
            image=images,
            height=vae_h,
            width=vae_w,
        ).unsqueeze(
            2
        )  # (B*num_cond_per_target,C,F,H,W)

        num_channels = pipeline.transformer.config.in_channels // 4
        eps_latents, img_latents = pipeline.prepare_latents(
            images=torch.cat([vae_targets, vae_images], dim=0),  # (B*num_cond_per_target + B,C,F,H,W)
            batch_size=(num_cond_per_target + 1) * batch_size,
            num_channels_latents=num_channels,
            height=vae_target_h,
            width=vae_target_w,
            device=device,
            dtype=weight_dtype,
            generator=generator,
        )
        eps_latents = eps_latents[:1]
        tgt_latents = img_latents[:batch_size]
        img_latents = img_latents[batch_size:]

        image_shapes = [
            [
                (1, vae_target_h // multiple_of, vae_target_w // multiple_of),
                (1, vae_h // multiple_of, vae_w // multiple_of),
                (1, vae_h // multiple_of, vae_w // multiple_of),
            ]
        ] * batch_size

        return LatentOutputs(
            noise_latents=eps_latents,
            target_latents=tgt_latents,
            condition_latents=img_latents,
            others={
                "image_shapes": image_shapes,
                "target_height": target_h,
                "target_width": target_w,
            },
        )

    @staticmethod
    def sample_latents(pipeline: QwenImageEditPlusPipeline, batch: dict[str, Any], **kwargs) -> SampleOutputs:
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

        def _get_sigmas(timesteps: torch.Tensor, latent_ndim: int = 4) -> torch.Tensor:
            sched_sigmas = scheduler.sigmas.to(device)
            sched_timesteps = scheduler.timesteps.to(device)
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
    def denoise_forward(pipeline: QwenImageEditPlusPipeline, batch: dict[str, Any], **kwargs) -> DenoiseOutputs:
        condition_outputs: ConditionOutputs = kwargs["condition_outputs"]
        latent_outputs: LatentOutputs = kwargs["latent_outputs"]
        sample_outputs: SampleOutputs = kwargs["sample_outputs"]
        attention_kwargs: dict[str, Any] = kwargs.get("attention_kwargs", {})

        B, L, C = latent_outputs.noise_latents.shape
        num_conditions = latent_outputs.condition_latents.shape[0]
        num_cond_per_target = num_conditions // B

        condition_latents = latent_outputs.condition_latents.view(num_cond_per_target, B, L, C)
        condition_latents = [condition_latents[i, ...] for i in range(num_cond_per_target)]
        latents = torch.cat([sample_outputs.noisy_latents] + condition_latents, dim=1)

        pred_v = pipeline.transformer(
            hidden_states=latents,
            encoder_hidden_states=condition_outputs.prompt_embeds,
            encoder_hidden_states_mask=condition_outputs.prompt_embeds_mask,
            timestep=sample_outputs.timesteps / 1000,
            img_shapes=latent_outputs.others["image_shapes"],
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        pred_v = pred_v[:, : sample_outputs.noisy_latents.shape[1]]
        return DenoiseOutputs(predictions=pred_v)

    @staticmethod
    def criterion_fn(pipeline: QwenImageEditPlusPipeline, batch: dict[str, Any], **kwargs) -> CriterionOutputs:
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
    def forward_step(
        pipeline: QwenImageEditPlusPipeline,
        batch: dict[str, Any],
    ) -> ForwardOutputs | ConditionOutputs:
        device = pipeline.transformer.device
        weight_dtype = pipeline.transformer.dtype

        QwenImageEditPlusForwardHandler.check_inputs(pipeline, batch)
        condition_outputs = QwenImageEditPlusForwardHandler.encode_conditions(
            pipeline,
            batch,
            device=device,
        )
        latent_outputs = QwenImageEditPlusForwardHandler.encode_latents(
            pipeline,
            batch,
            device=device,
            weight_dtype=weight_dtype,
            vae_max_resolution=480*480,
        )
        sample_outputs = QwenImageEditPlusForwardHandler.sample_latents(
            pipeline,
            batch,
            target_latents=latent_outputs.target_latents,
            noise_latents=latent_outputs.noise_latents,
        )
        denoise_outputs = QwenImageEditPlusForwardHandler.denoise_forward(
            pipeline,
            batch,
            condition_outputs=condition_outputs,
            latent_outputs=latent_outputs,
            sample_outputs=sample_outputs,
        )
        criterion_outputs = QwenImageEditPlusForwardHandler.criterion_fn(
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
