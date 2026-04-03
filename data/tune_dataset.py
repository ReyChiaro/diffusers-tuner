import json

import importlib
from torch.utils.data import Dataset
from typing import Callable, Any, TypedDict, Annotated, Literal

from data.key_processors import default_key_processor


type DataFileFormat = Annotated[Literal["jsonl"], "Supported format for data file."]


class KeySchemaType(TypedDict):
    r"""
    Example: {
        "model_key": "image",
        "data_key": ("imaeg_path_1", "image_path_2"),
        "processor": ... # A function that maps "data_key" to "model_key"
    }
    """

    model_key: str
    data_key: str | tuple[str]
    processor: str | Callable[[Any], Any] | None


class TuneDataset(Dataset):

    def __init__(
        self,
        key_schemas: dict[str, KeySchemaType],
        data_file: str,
        data_file_type: DataFileFormat = "jsonl",
    ):
        r"""
        :param data_file: Lines of data samples, default JSONL format.
        """
        super().__init__()

        key_schemas: list[KeySchemaType] = list(s for s in key_schemas.values())
        self.key_schemas: list[KeySchemaType] = []
        for _schema in key_schemas:
            schema = _schema

            # Handle key processor
            processor = _schema.get("processor", None)
            if processor is None:
                processor = default_key_processor
            elif isinstance(processor, str):
                module_path = ".".join(processor.split(".")[:-1])
                processor_name = processor.split(".")[-1]
                processor = getattr(importlib.import_module(module_path), processor_name)
            schema["processor"] = processor

            # Handle data key(s)
            data_key = _schema.get("data_key", None)
            if data_key is None:
                raise KeyError(f"data_key should not be None.")
            if isinstance(data_key, str):
                schema["data_key"] = (data_key,)

            self.key_schemas.append(schema)

        # Keys passed to DataLoader
        self.model_keys = [s["model_key"] for s in self.key_schemas]

        # Load samples from data_file
        self.data_file_type = data_file_type

        load_fn = getattr(
            importlib.import_module("data.data_utils"),
            f"load_{self.data_file_type}_file",
        )
        self.samples = load_fn(data_file)

        self.num_samples = len(self.samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % self.num_samples]
        return_dict = {k: None for k in self.model_keys}

        for schema in self.key_schemas:
            model_key = schema["model_key"]
            return_dict[model_key] = schema["processor"](*[sample[dk] for dk in schema["data_key"]])

        return return_dict
