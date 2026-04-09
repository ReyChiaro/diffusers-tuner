import random
import importlib

from torch.utils.data import Dataset, Sampler
from typing import Callable, Any, TypedDict, Annotated, Literal

from loguru import logger
from data.data_utils import load_json_file

type DataFileFormat = Annotated[Literal["jsonl", "json"], "Supported format for data file."]
type PostProcessor = Callable[[dict[str, Any], dict[str, Any], int], dict[str, Any] | None]


class KeySchemaType(TypedDict):
    r"""
    Example: {
        "model_key": "image",
        "data_key": ("image_path_1", "image_path_2"),
        "processor": <callable> # A function that maps "data_key" to "model_key",
    }
    """

    model_key: str
    data_key: str | tuple[str]
    processor: str | Callable[[Any], Any] | None = None


def _resolve_callable(callable_or_path: str | Callable[..., Any]) -> Callable[..., Any]:
    if isinstance(callable_or_path, str):
        module_path = ".".join(callable_or_path.split(".")[:-1])
        callable_name = callable_or_path.split(".")[-1]
        return getattr(importlib.import_module(module_path), callable_name)
    return callable_or_path


class SchemaDataset(Dataset):

    def __init__(
        self,
        key_schemas: dict[str, KeySchemaType],
        collate_fn: str | None = None,
        post_processors: list[str | PostProcessor] | None = None,
    ):
        r"""
        :param data_file: Lines of data samples, default JSONL format.
        """
        super().__init__()

        self.collate_fn = collate_fn
        key_schemas: list[KeySchemaType] = list(s for s in key_schemas.values())
        self.key_schemas: list[KeySchemaType] = []
        for _schema in key_schemas:
            schema = _schema

            # Handle key processor
            processor = _schema.get("processor", None)
            if processor is None:
                processor = None
            elif isinstance(processor, str):
                processor = _resolve_callable(processor)
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
        self.post_processors = self._normalize_post_processors(post_processors)

    def _normalize_post_processors(
        self,
        post_processors: list[str | PostProcessor] | None,
    ) -> list[PostProcessor]:
        if post_processors is None:
            return []

        normalized_post_processors = []
        for post_proc in post_processors:
            normalized_post_processors.append(_resolve_callable(post_proc))
        return normalized_post_processors

    def load_samples(self):
        return [], 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % self.num_samples]
        return_dict = {k: None for k in self.model_keys}

        for schema in self.key_schemas:
            model_key = schema["model_key"]
            return_dict[model_key] = schema["processor"](*[sample[dk] for dk in schema["data_key"]])

        for post_proc in self.post_processors:
            updated = post_proc(return_dict, sample, index)
            if updated is not None:
                return_dict = updated

        return return_dict


class TuneDataset(SchemaDataset):

    def __init__(
        self,
        key_schemas: dict[str, KeySchemaType],
        data_file: str,
        data_file_type: DataFileFormat = "jsonl",
        collate_fn: str | None = None,
        post_processors: list[str | PostProcessor] | None = None,
    ):
        r"""
        :param data_file: Lines of data samples, default JSONL format.
        """
        super().__init__(key_schemas, collate_fn, post_processors)

        self.data_file = data_file
        self.data_file_type = data_file_type
        self.samples, self.num_samples = self.load_samples()

    def load_samples(self) -> tuple[list[dict[str, Any]], int]:
        # Load samples from data_file

        load_fn = getattr(
            importlib.import_module("data.data_utils"),
            f"load_{self.data_file_type}_file",
        )
        samples = load_fn(self.data_file)
        num_samples = len(samples)
        return samples, num_samples


class TuneBucketDataset(SchemaDataset):

    def __init__(
        self,
        key_schemas: dict[str, KeySchemaType],
        bucket_data_file: str,
        bucket_data_file_type: DataFileFormat = "json",
        collate_fn: str | None = None,
        post_processors: list[str | PostProcessor] | None = None,
    ):
        super().__init__(key_schemas, collate_fn, post_processors)

        # Load samples from data_file
        self.data_file = bucket_data_file
        self.data_file_type = bucket_data_file_type

        self.samples, self.num_samples = self.load_samples()

    def load_samples(self) -> tuple[list[dict[str, Any]], int]:
        load_fn = getattr(
            importlib.import_module("data.data_utils"),
            f"load_{self.data_file_type}_file",
        )

        # Bucket ID -> list of samples
        bucket_samples = load_fn(self.data_file)

        # Flattened samples
        samples = []
        buckets = {}

        # Bucket ID -> sample index in flattened samples
        s_idx = 0
        for b_idx, b_samples in bucket_samples.items():
            buckets[int(b_idx)] = []
            for sample in b_samples:
                samples.append(sample)
                buckets[int(b_idx)].append(s_idx)
                s_idx += 1

        self.buckets = buckets
        return samples, len(samples)


class KeySchema(TypedDict):

    data_key: str | list[str]
    key_processor: str | None


class DatasetSchema(Dataset):

    def __init__(self, data_file: str, key_schemas: dict[str, KeySchema], collate_fn: str | None):
        super().__init__()

        self.data_file = data_file
        self.model_keys = list(key_schemas.keys())

        self.schema = {
            mk: {
                "data_key": key_schemas[mk]["data_key"],
                "key_processor": self._instantiate_key_processor(key_schemas[mk]["key_processor"]),
            }
            for mk in self.model_keys
        }
        self.collate_fn = self._instantiate_collate_fn(collate_fn)

    def _instantiate_key_processor(self, key_processor: str | None) -> Callable[[Any], Any]:
        r"""
        :param key_processor: Format of key_processor is `<data_module>.<key_processor>`.
            Then this method will import the key processor from data.<data_module>.key_processors.<key_processor> and return it.
        """
        if key_processor is None:
            # Return default processor
            module = importlib.import_module(f"data.key_processors")
            processor = getattr(module, "default_processor")
            return processor

        data_module = key_processor.split(".")[0]
        processor_name = key_processor.split(".")[1]
        module = importlib.import_module(f"data.{data_module}.key_processors")
        processor = getattr(module, processor_name)
        return processor

    def _instantiate_collate_fn(self, collate_fn: str | None) -> Callable[[list[dict[str, Any]]], Any] | None:
        r"""
        :param collate_fn: Format of collate_fn is `<data_module>.<collate_fn>`.
            Then this method will import the collate function from data.<data_module>.collate_fns.<collate_fn> and return it.
        """
        if not collate_fn:
            return None

        data_module = collate_fn.split(".")[0]
        collate_name = collate_fn.split(".")[1]
        module = importlib.import_module(f"data.{data_module}.collate_fns")
        collate_function = getattr(module, collate_name)
        return collate_function

    def load_samples(self) -> tuple[list[dict[str, Any]], int]:
        logger.warning(
            f"load_samples maybe not be implemented yet, use default, which may cause error or undesired results."
        )
        return [], 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % self.num_samples]
        model_sample = {}
        for mk in self.model_keys:
            # 1. str: return str
            # 2. image path: return Image
            # 3. list of image path: return list of Image
            # 4. user defined other conditions
            model_sample[mk] = self.schema[mk]["key_processor"](**{dk: sample[dk] for dk in self.schema[mk]["data_key"]})
        return model_sample


class DiffusersTunerDataset(DatasetSchema):

    def __init__(self, data_file, key_schemas, collate_fn, bucket_dataset: bool = False):
        r"""
        :param bucket_dataset (bool): Default: False
        """
        self.bucket_dataset = bucket_dataset
        self.buckets = None

        super().__init__(data_file, key_schemas, collate_fn)
        self.samples, self.num_samples = self.load_samples()

    def load_samples(self):
        r"""
        If bucket dataset is enabled, then the provided dataset file should be {<bucket_id>: <data_item>}. If not, the file is list of data items. Both conditions use JSON format.
        """
        samples = []
        if self.bucket_dataset:
            # Bucket ID -> list of samples
            bucket_samples = load_json_file(self.data_file)

            # Flattened samples
            buckets = {}

            # Bucket ID -> sample index in flattened samples
            s_idx = 0
            for b_idx, b_samples in bucket_samples.items():
                buckets[int(b_idx)] = []
                for sample in b_samples["dataset"]:
                    samples.append(sample)
                    samples[-1].update({"aspect_ratio": b_samples["aspect_ratio"]})
                    buckets[int(b_idx)].append(s_idx)
                    s_idx += 1

            self.buckets = buckets
        else:
            samples = load_json_file(self.data_file)
        print(samples)
        return samples, len(samples)


class BucketBatchSampler(Sampler):
    r"""
    An accelerate compatible batch sampler, support DDP training.
    Note that we should initialize accelerator with `dispatch_batches=False`.
    """

    def __init__(
        self,
        bucket_indices: dict[int, list[int]],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        seed: int = 42,
    ):
        r"""
        :param num_replicas (int, default 1): Set to `num_processes` or `ngpu`. Used to split
            batches into different processes in DDP training.
        """
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last

        self.batches = self._prepare_batches()

    def _prepare_batches(self):
        batches = []
        for _, sids in self.bucket_indices.items():
            random.seed(self.seed)
            random.shuffle(sids)

            for i in range(0, len(sids), self.batch_size):
                batch = sids[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        random.seed(self.seed)
        random.shuffle(batches)

        return batches[self.rank :: self.num_replicas]

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)
