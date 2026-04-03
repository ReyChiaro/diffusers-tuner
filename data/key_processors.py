from PIL import Image
from typing import Any

from data.data_utils import get_centercrop_params


default_key_processor = lambda *args: args[0]


def load_pil_from_path(*paths: str) -> Image.Image | list[Image.Image]:
    images = [Image.open(p).convert("RGB") for p in paths]
    return images[0] if len(images) == 1 else images

