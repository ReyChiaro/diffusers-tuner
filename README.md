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
> TODO

## Memory Saving
> TODO

### Offload text encoder
> TODO

## 📝 Roadmap

- [x] Full Tuning Loop Implementation
- [x] Hierarchical Configuration System
- [x] DDP & Accelerator Support
- [x] Evaluation Interfaces & Benchmarking
- [ ] [In progress] Support for more SOTA pipelines