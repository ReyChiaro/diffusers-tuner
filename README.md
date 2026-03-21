# Diffusers Tuner: A Simple Finetuner for 🤗[diffusers](https://github.com/huggingface/diffusers)

Do not want to re-write the whole pipelines? Hunger for configuration system? A suitable tuning framework? Do not know how to register and enable a LoRA for pipeliens? There it is, our diffusers tuner solves provides a tuning framework for diffusers. This repository implements a tuning pipeline including data preparations, `__call__` like tune step interface, and configuration system with DDP support.


# 🚀 Quick Start

Just `git clone` or download zip of this repo. We provide both `pyproject.toml` and `requirements.txt` for `uv` as well `pip` users.

For `uv`, the following command is all you need.
```sh
uv sync
```
And for `pip` just 
```sh
pip install -r requirements.txt
```

> Note that this repo is based on `python3.12` with `torch>=2.10` and `diffusers>=0.37.0`, if you want to use it for lower version, maybe you can change the required lowest version in `pyproject.toml` and `requirements.txt` (We use these versions for we provide a `QwenImageEdit` demo in this repo).

The folder structure is
```sh
.
│   # Configs
├── configs
│   ├── adapter
│   │   └── lora.yaml
│   ├── dataset
│   │   ├── data_module.yaml
│   │   ├── eval_set.yaml
│   │   └── tune_set.yaml
│   ├── loss
│   │   └── flow_matching.yaml
│   ├── pipeline
│   │    └── qwenimage_edit_plus.yaml
│   └── main.yaml
│
│   # Main codebase
├── adapters
│   ├── __init__.py
│   ├── lora.py
│   └── utils.py
├── data
│   ├── __init__.py
│   ├── data_module.py
│   └── style_transfer_dataset.py
├── LICENSE
├── main.py
├── pipelines
│   └── pipeline_qwenimage_edit_plus.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── tuner.py
└── uv.lock
```

## 🐈 How to use

If everything is prepared, just launch the tuning by
```sh
python main.py
```
The outputs and saved checkpoints can be found in `outputs` (you can modify the path in `configs/main.yaml`).

For an advanced usage, you can contunue read the following configuration definitions, which can be modified by runtime command line or files.

### 🧩 Configs

Files in `configs` provide configurations, all syntax can be found in [hydra](https://github.com/facebookresearch/hydra). But for a simple case, we have delivered a demo for [QwenImageEdit](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).

**adapter**: Defines the finetune adapter, we provide `lora.yaml` for instance. Note that there are little differences between ours and [peft](https://github.com/huggingface/peft). We NOT implement some complex functions, making tuning simple, only `rank` and `target_modules` are what we really care about for the most of time.

**dataset**: Datasets used for tuning should be prepared by this configuration. Normally, we care about following attributes in one sample:

- `images`: List of images tensors, which makes our framework usable for multiple images inputs.
- `prompts`: List of `str` prompts, optional if `prompt_embeds` and `prompt_embeds_mask` are given. Note that if prompts are given, no matter whether embeddings and masks are given, we will encode the prompts to generate new features.
- `targets`: Tensor, the generated targets.
- `prompt_embeds` and `prompt_embeds_mask`: Tensor, the text encoder results, required if NO prompts are specified.
- `negative_prompts`: List of `str`.
- `negative_prompt_embeds` and `negative_prompt_embeds_mask`: Tensor, text encoder encoded features for `negative_prompts`.

**loss**: Some arguments used to calculate loss.

**pipeline**: Defines which module (module names can be got by `pipeline.components`) shuold the adapter attatch to and the pretrained pipeline path.

**main**: Aggregate configurations from above and defines training parameters as well as accelerator and tuner arguments.


We give a demo for modifying/adding the configuration runtime, other usage can be found in [hydra](https://github.com/facebookresearch/hydra):
```sh
python main.py \
    data_module.tune_batch_size=4 \   # modify batchsize
    adapter.rank=32 \                 # modify LoRA rank
    +additional_args="hello word" \   # add addtional key-value pair
```


# 📝 TODO

Though our tuner complete tuning loop, there are also some future features we want to implement:

- ✅ Tuning loop
- ✅ Configuration system
- ✅ Launching script
- ☑️ Evaluation interfaces
- ☑️ Support more pipelines