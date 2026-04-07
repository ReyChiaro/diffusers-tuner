import torch
import torchvision.transforms.functional as T

from typing import Any

from data.data_utils import get_centercrop_params


class PairedCenterCrop:
    def __init__(
        self,
        model_image_key: str = "image",
        model_target_key: str = "target",
        raw_aspect_ratio_key: str = "target_aspect_ratio",
    ):
        self.model_image_key = model_image_key
        self.model_target_key = model_target_key
        self.raw_aspect_ratio_key = raw_aspect_ratio_key

    def __call__(
        self,
        model_sample: dict[str, Any],
        raw_sample: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        
        image = model_sample[self.model_image_key]
        target: torch.Tensor = model_sample[self.model_target_key]

        normed_wh: tuple[int, int] = raw_sample[self.raw_aspect_ratio_key]
        aspect_ratio = normed_wh[0] / normed_wh[1]

        target_h = int(target.shape[-1] / aspect_ratio)
        target_w = int(target.shape[-2] * aspect_ratio)
        i, j, h, w = get_centercrop_params((target.shape[-1], target.shape[-2]), [target_w, target_h])
        model_sample[self.model_target_key] = T.resized_crop(target, i, j, h, w, [target_h, target_w])

        input_images = []
        if isinstance(image, torch.Tensor):
            image = [image]
        for img in image:
            img_h = int(img.shape[-1] / aspect_ratio)
            img_w = int(img.shape[-2] * aspect_ratio)
            i, j, h, w = get_centercrop_params((img.shape[-1], img.shape[-2]), [img_w, img_h])
            input_images.append(T.resized_crop(img, i, j, h, w, [target_h, target_w]))

        model_sample[self.model_image_key] = input_images[0] if len(input_images) == 0 else input_images

        return model_sample
