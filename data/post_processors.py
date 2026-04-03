import torchvision.transforms.functional as T

from PIL import Image
from typing import Any, Iterable

from data.data_utils import get_centercrop_params


class PairedCenterCrop:
    def __init__(
        self,
        image_key: str = "image",
        target_key: str = "target",
        aspect_ratio_key: str = "target_aspect_ratio",
    ):
        self.image_key = image_key
        self.target_key = target_key
        self.aspect_ratio_key = aspect_ratio_key

    def __call__(
        self,
        model_sample: dict[str, Any],
        raw_sample: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        image = model_sample[self.image_key]
        target: Image.Image = model_sample[self.target_key]
        
        normed_wh: tuple[int, int] = raw_sample[self.aspect_ratio_key]
        aspect_ratio = normed_wh[0] / normed_wh[1]

        target_h = int(target.width / aspect_ratio)
        target_w = int(target.height * aspect_ratio)

        i, j, h, w = get_centercrop_params(target.size, [target_w, target_h])
        model_sample[self.target_key] = T.crop(target, i, j, h, w)

        if isinstance(image, Image.Image):
            image = [image]
        image = [T.crop(img, i, j, h, w) for img in image]
        model_sample[self.image_key] = image[0] if len(image) == 0 else image

        return model_sample
