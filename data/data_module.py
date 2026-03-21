import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Dict, Any, Tuple


class DataModule:

    def __init__(
        self,
        tune_set: Dataset,
        tune_batch_size: int,
        tune_num_workers: int,
        eval_set: Optional[Dataset] = None,
        eval_batch_size: Optional[int] = None,
        eval_num_workers: Optional[int] = None,
        enable_ddp: bool = False,
    ):
        self.tune_set = tune_set
        self.eval_set = eval_set
        self.tune_batch_size = tune_batch_size
        self.eval_batch_size = eval_batch_size
        self.tune_num_workers = tune_num_workers
        self.eval_num_workers = eval_num_workers
        self.enable_ddp = enable_ddp

    def basic_collate_fn(self, batch: Tuple[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = []
        negative_prompts = []
        targets = []
        prompt_latents = []
        prompt_latents_mask = []
        negative_prompt_latents = []
        negative_prompt_latents_mask = []

        num_images_per_prompt = len(batch[0]["images"])
        images = [[]] * num_images_per_prompt
        for sample in batch:
            input_images = sample["images"]  # List of Tensor
            for i in range(num_images_per_prompt):
                images[i].append(input_images[i])
            prompts.append(sample["prompts"])
            targets.append(sample["targets"])
            negative_prompts.append(sample["negative_prompts"])
            prompt_latents.append(sample["prompt_latents"])
            prompt_latents_mask.append(sample["prompt_latents_mask"])
            negative_prompt_latents.append(sample["negative_prompt_latents"])
            negative_prompt_latents_mask.append(sample["negative_prompt_latents_mask"])

        targets = torch.stack(targets)
        images = [torch.stack(batch_imgs) for batch_imgs in images]
        if prompt_latents[0] is not None:
            prompt_latents = torch.stack(prompt_latents)
        if prompt_latents_mask[0] is not None:
            prompt_latents_mask = torch.stack(prompt_latents_mask)
        if negative_prompt_latents[0] is not None:
            negative_prompt_latents = torch.stack(negative_prompt_latents)
        if negative_prompt_latents_mask[0] is not None:
            negative_prompt_latents_mask = torch.stack(negative_prompt_latents_mask)

        return {
            "images": images,
            "prompts": prompts,
            "negative_prompts": negative_prompts,
            "targets": targets,
            "prompt_latents": prompt_latents,
            "prompt_latents_mask": prompt_latents_mask,
            "negative_prompt_latents": negative_prompt_latents,
            "negative_prompt_latents_mask": negative_prompt_latents_mask,
        }

    @property
    def tune_loader(self):
        tune_sampler = DistributedSampler(self.tune_set) if self.enable_ddp else None
        return DataLoader(
            self.tune_set,
            batch_size=self.tune_batch_size,
            shuffle=tune_sampler is None,
            sampler=tune_sampler,
            num_workers=self.tune_num_workers,
            collate_fn=self.basic_collate_fn,
        )

    @property
    def eval_loader(self):
        eval_sampler = DistributedSampler(self.eval_set) if self.enable_ddp else None
        return DataLoader(
            self.eval_set,
            batch_size=self.eval_batch_size,
            shuffle=eval_sampler is None,
            sampler=eval_sampler,
            num_workers=self.eval_num_workers,
            collate_fn=self.basic_collate_fn,
        )
