# 🎨 Diffusers Tuner: A Lightweight, Configuration-Driven, Flexible Fine-Tuning Framework for [🤗 Diffusers](https://github.com/huggingface/diffusers)

> This project is still in progress. Welcome to contribute and provide PR 🤓☝️

Diffusers Tuner is designed to eliminate the overhead of fine-tuning diffusion models. Tired of rewriting training pipelines from scratch? Struggling with complex LoRA injections or messy configuration management?

Our framework provides a streamlined, "plug-and-play" experience for the diffusers ecosystem, implementing a complete pipeline including data preparation, a tuning interface, and a robust configuration system with native DDP support.

## 🎥 News

[2026/04/08] [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) pipeline is supported. \
[2026/03/18] [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) is added to our framework. \
[2026/03/17] The first commit is created!

## ✨ Key Features

- Zero-Friction Adaptation: Tune existing diffusers pipelines without refactoring the core logic.
- Config-First Workflow: Powered by [Hydra](https://hydra.cc/) for flexible, hierarchical configuration management.
- Dataset Schema: Adapts to various format of dataset by defining the dataset schema.
- Bucket Batched Dataset Training: Support bucket dataset training for larger batchsize with different resolutions.
- Lightweight LoRA: A simplified LoRA implementation that focuses on what matters (the `rank` and `target_modules`), without the complexity of heavy libraries.
- Modern Model Support: Includes a specialized demo for [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) and [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev).

## 🚀 Quick Start

### Installation

**Requirements**: This repository targets `Python 3.12, PyTorch 2.10+, and Diffusers 0.37.0+` to support advanced features like QwenImageEdit.

We recommend `uv` for lightning-fast dependency management, but standard `pip` works perfectly too:

```sh
git clone https://github.com/your-username/diffusers-tuner.git
cd diffusers-tuner

# Using uv (Recommended)
uv sync

# Or using pip
# pip install -r requirements.txt
```

### Launch fine-tuning

Start your training session with `accelerate` and `hydra`:

```sh
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --gpu_ids 0 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    tune.py --config-name tune \
    tune.project_name=tune-qwenimage \
    tune.max_tuning_steps 10 \
    tune.save_steps 5 \
    tune.eval_steps 5 \
    tune.mixed_precision bf16 \
    pipeline=qwenimage_edit_plus \
    "pipeline.load_modules=[scheduler,processor,tokenizer,vae,transformer,text_encoder]" \
    "pipeline.tune_modules=[transformer]" \
    adapter=lora \
    adapter.rank=32 \
    "adapter.target_modules=[attn.to_q,attn.to_k,attn.to_v]" \
    data_cfgs.tune_batch_size=1 \
    data_cfgs.enable_bucket_data=ture \
    dataset=dummy \
```

After doing this, you will see there is a new folder named `outputs` where there is a subfolder named by `tune.project_name`. For each time running the same experiment with same project name, it will create a subfolder named by the timestamp. If you want to recover a configuration of former experiments, check `<output_dir>/<project_name>/<timestamp>/hydra-configs/config.yaml` and just run following lines to use the same configurations to start a experiment. (You can check [Hydra](https://hydra.cc/) for more fancy usages.)

```sh
accelerate launch tune.py \
    --config-path <output_dir>/<project_name>/<timestamp>/hydra-configs \
    --config-name config \
```

Note that the framework will download the pipeline from huggingface, you can modify the model path if you have downloaded it by adding configuration `pipeline.pretrained_model_name_or_path`. Check [Configurations](#configurations) for more personalized configurations.

> Device requirements: For above demo script, the GPU memory should be at least 104G with image resolution $512\times 512$ and `bfloat16` weights. A effective way to save memory is to offload the text encoder and prepare prompts embeddings ahead, see [Memory Saving](#memory-saving) for more information.


## Project Structure

A simple yet effective structure helps to understand the core of our framework.

```Plaintext
.
├── configs/             # Hydra configuration center (Adapter, Dataset, Loss, Pipeline, etc.)
├── adapters/            # Lightweight adapter implementations
├── data/                # Datasets and schema
├── pipelines/           # Pipeline and forward handler
├── diffusers_tuner/     # The core tuning loop engine
├── tune.py              # Training
├── inference.py         # Inference
└── encode_prompt.py     # Text encoder embeddings
```


## Configurations

This project is fully **Hydra-driven**, and all configurations live under `configs/`.
You can compose configs by selecting groups and override any field from the CLI.

### Config Groups

- `pipeline`: Which diffusers pipeline to use and which components to load/tune.
- `adapter`: LoRA configs (rank, target modules, adapter name, etc.).
- `dataset`: Dataset schema and collate function definition.
- `loss`: Loss choice.
- `tune`: Training configuration.

### Common Overrides

```sh
# Pipeline selection and module control
pipeline=qwenimage_edit_plus \
"pipeline.load_modules=[scheduler,processor,tokenizer,vae,transformer,text_encoder]" \
"pipeline.tune_modules=[transformer]" \

# Adapter controls
adapter=lora \
adapter.rank=32 \
"adapter.target_modules=[attn.to_q,attn.to_k,attn.to_v]" \
adapter.adapter_name=my_lora \

# Training controls
tune.max_tuning_steps=1000 \
tune.save_steps=100 \
tune.eval_steps=100 \
tune.mixed_precision=bf16 \

# Dataset and data loader
dataset=dummy \
dataset.data_file=/path/to/your/data.json \
data.tune_batch_size=1 \
data.tune_num_workers=4
```


## Framework Architecture

The design separates **pipeline logic**, **training loop**, and **data schema**, so you can swap
components without totally rewriting training code (in fact, you should also code some forward logic by inheriting the forward handler).

### Forward Handler

Each pipeline has a matching **ForwardHandler** that defines the training-time logic:

- `encode_conditions`: prompts or precomputed embeddings
- `encode_latents`: image/target to latent space
- `sample_latents`: diffusion timestep sampling
- `denoise_forward`: model forward pass
- `criterion_fn`: loss calculation
- `forward_step`: composition of the above

This keeps training logic pipeline-specific but **API-consistent** across models.

### Lightweight Adapter System

Adapters are implemented in `adapters/utils.py`:

- `register_adapter`: wraps `nn.Linear` as `AdapterManager`
- `add_adapter`: adds LoRA params to matched modules
- `activate_adapter(s)`: choose which adapter(s) to apply in forward
- `enable_adapter`: toggles training (`requires_grad`) only

This makes adapter activation independent from training, and supports multi-adapter mixing
by weights.

> To be honest, I usually fails to fine-tune my model with `peft` in `diffusers`... So I make the fine-tune adapter simpler to help me clearify what is really going on.

### Tuner Engine

`diffusers_tuner/tuner.py` wraps **Accelerate** and exposes:

- Full DDP-compatible training loop
- Optional evaluation hooks
- Safe checkpoint saving for adapter weights

The trainer takes any `TunePipeline` + `DatasetSchema` pair and runs end-to-end.


## Detailed Usage

### Training

The main entry is `tune.py`. The compulsive configurations are `pipeline` and `dataset`, you should always provide the two fields to start a fine-tuning task (But note that the adapter tuning modules are empty be default, maybe you shoud pass some modules to it).

```sh
accelerate launch tune.py pipeline=qwenimage_edit_plus dataset=dummy
```

### Inference

Use `inference.py` and provide `adpt_checkpoint`:

```sh
python inference.py \
    pipeline=qwenimage_edit_plus \
    adapter=lora \
    "adapter.target_modules=[attn.to_q,attn.to_k,attn.to_v]" \
    +adpt_checkpoint=/path/to/your.safetensors \
    +prompt="A cinematic scene of a robot painter"
```

### Multi-Adapter Merge

You can load multiple adapters and activate them with weights:

```python
from adapters.utils import activate_adapters

activate_adapters(
    model,
    {"style_a": 1.0, "style_b": 0.6},
    clear_others=True,
)
```

### Dataset Format

The `dataset` config defines a **schema** that maps your JSON fields to model inputs.

Example (simplified):

```yaml
key_schemas:
  images:
    data_key: ["image","mask"]
    key_processor: null
  prompts:
    data_key: "prompt"
    key_processor: null
  targets:
    data_key: "target"
    key_processor: null
collate_fn: null
```

The `key_processor` and `collate_fn` are default or self-defined methods.

- key processor: Used to load JSON dataset, convert data keys in data file to model keys in batch,
- collate function: Used to post-process the data in batch level.

For a custom dataset, you can create a folder named `my_custom_dataset` (or anything you like) under `data` folder, and then define your collate functions and key processors in there. Finally, you can use these custom methods by filling the configurations:

```yaml
key_processor: my_custom_dataset.custom_key_processor
collate_fn: my_custom_dataset.custom_collate_fn
```

### Bucket Dataset

A bucket dataset helps to batch your training data especially for different aspect ratios.
If `bucket_dataset: true` is set, your `data_file` should be a JSON object:

```json
{
  "0": { "aspect_ratio": [1, 1], "dataset": [ ... ] },
  "1": { "aspect_ratio": [3, 4], "dataset": [ ... ] },
  "2": { "aspect_ratio": [9, 16], "dataset": [ ... ] }
}
```

Each item in `dataset` is a normal sample. The default `collate_fn` can resize to the bucket size.


## Memory Saving

Large image-edit pipelines are GPU-hungry. The framework provides a few built-in options:

- **Offload text encoder** and precompute prompt embeddings
- **Only load required modules** in `pipeline.load_modules`
- **Use bf16** mixed precision

### Recommended Strategy

1. Precompute prompt embeddings with `encode_prompt.py`
2. Remove `text_encoder` from `pipeline.load_modules`
3. Feed `prompt_embeds` + `prompt_embeds_mask` during training (Note that you can modify the data configuration to achieve this feature)


## 📝 Roadmap

- [x] Full Tuning Loop Implementation
- [x] Hierarchical Configuration System
- [x] DDP & Accelerator Support
- [x] Evaluation Interfaces & Benchmarking
- [ ] [In progress] Support for more SOTA pipelines
