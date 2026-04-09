import torch
import torchvision.transforms.functional as T

from typing import Any


def flatten_and_resize(batch: list[dict[str, Any]]) -> dict[str, Any]:
    r"""resize all images to the same size as targets"""

    def flatten_lists(data):
        def is_list_of_str(lst):
            return isinstance(lst, list) and all(isinstance(i, str) for i in lst)

        result = []
        for item in data:
            if isinstance(item, list) and not is_list_of_str(item):
                result.extend(flatten_lists(item))
            else:
                result.append(item)
        if isinstance(result[-1], torch.Tensor):
            result = torch.stack(result, dim=0)
        return result

    keys = list(batch[0].keys())
    batch = {k: flatten_lists([b[k] for b in batch]) for k in keys}

    targets = batch["targets"]
    h, w = targets.shape[-2:]
    images = T.resize(batch["images"], [h, w])
    batch["images"] = images
    return batch
