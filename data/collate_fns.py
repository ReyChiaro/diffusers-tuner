import importlib

from loguru import logger


class CollateFn:

    def __init__(self, collate_fn_path: str):
        try:
            module = importlib.import_module(".".join(collate_fn_path.split(".")[:-1]))
            self.collate_fn = getattr(module, collate_fn_path.split(".")[-1])
            logger.info(f"Use collate fn found in {collate_fn_path}")
        except:
            self.collate_fn = lambda x: x

    def __call__(self, batch):
        return self.collate_fn(batch)
