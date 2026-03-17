import torch.nn as nn

from adapters.utils import AdapterManager


class LoRA(AdapterManager):

    def __init__(self, base_model: nn.Module):
        super().__init__(base_model)
