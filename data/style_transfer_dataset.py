import os
import torchvision.transforms.transforms as T

from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional


class StyleTransferDataset(Dataset):

    def __init__(
        self,
        content_dir: str,
        style_dir: str,
        result_dir: str,
        image_height: int,
        image_width: int,
        requires_prompt: bool = False,
        prompt_latents_dir: Optional[str] = None,
    ):
        r"""
        Naming rule:
            content images: content_dir/<4_bit_number>.jpg
            style images: style_dir/<4_bit_number>.jpg
            result images: result_dir/<cnt_id>_<sty_id>.jpg
            prompt latents: prompt_latents_dir/<cnt_id>_<sty_id>.safetensors
        """
        super().__init__()

        self.cnt_dir = content_dir
        self.sty_dir = style_dir
        self.res_dir = result_dir
        self.prompt_latents_dir = prompt_latents_dir

        self.prompt = "Transfer the style of image 1 to the style of image 2."

        # Split image ids
        self.samples = []
        for res in os.listdir(self.res_dir):
            res_suff = Path(res).suffix
            res_name = Path(res).stem
            parts = res_name.split("_")
            cnt_id = parts[0]
            sty_id = parts[1]

            self.samples.append(
                {
                    "images": [cnt_id + res_suff, sty_id + res_suff],
                    "prompts": self.prompt if requires_prompt else None,
                    "targets": res,
                    "prompt_latents_and_mask": None,
                }
            )

            if self.prompt_latents_dir is not None and os.path.exists(self.prompt_latents_dir):
                prompt_latents_name = f"{cnt_id}_{sty_id}.safetensors"
                self.samples[-1]["prompt_latents_and_mask"] = prompt_latents_name
        self.num_samples = len(self.samples)

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((image_height, image_width), T.InterpolationMode.BILINEAR),
            ]
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor]:
        index = index % self.num_samples
        sample = self.samples[index]
        prompt = sample["prompts"]
        cnt_name = sample["images"][0]
        sty_name = sample["images"][1]
        res_name = sample["targets"]
        prompt_latents_name = sample["prompt_latents_and_mask"]

        cnt_image = Image.open(os.path.join(self.cnt_dir, cnt_name)).convert("RGB")
        sty_image = Image.open(os.path.join(self.sty_dir, sty_name)).convert("RGB")
        res_image = Image.open(os.path.join(self.res_dir, res_name)).convert("RGB")

        cnt_image = self.transform(cnt_image)
        sty_image = self.transform(sty_image)
        res_image = self.transform(res_image)

        prompt_latents = None
        prompt_latents_mask = None

        if prompt_latents_name is not None:
            latent_and_mask = load_file(os.path.join(self.prompt_latents_dir, prompt_latents_name))
            prompt_latents = latent_and_mask["txt_latent"]
            prompt_latents_mask = latent_and_mask["txt_latent_mask"].long()

        return {
            "images": [cnt_image, sty_image],
            "prompts": prompt,
            "negative_prompts": None,
            "targets": res_image,
            "prompt_latents": prompt_latents,
            "prompt_latents_mask": prompt_latents_mask,
            "negative_prompt_latents": None,
            "negative_prompt_latents_mask": None,
        }
