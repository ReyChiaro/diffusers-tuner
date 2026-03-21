# 🎨 Diffusers Tuner: A Lightweight, Configuration-Driven Fine-tuning Framework for 🤗 [diffusers](https://github.com/huggingface/diffusers)

Diffusers Tuner is designed to eliminate the overhead of fine-tuning diffusion models. Tired of rewriting training pipelines from scratch? Struggling with complex LoRA injections or messy configuration management?

Our framework provides a streamlined, "plug-and-play" experience for the diffusers ecosystem, implementing a complete pipeline including data preparation, a high-level `__call__`-style tuning interface, and a robust configuration system with native DDP support.

## ✨ Key Features

- Zero-Friction Adaptation: Tune existing diffusers pipelines without refactoring the core logic.
- Config-First Workflow: Powered by Hydra for flexible, hierarchical configuration management.
- Lightweight LoRA: A simplified LoRA implementation that focuses on what matters—rank and target_modules—without the complexity of heavy libraries.
- Research-Ready: Built-in support for Multi-image inputs, Flow-matching loss, and academic-grade logging.
- Modern Model Support: Includes a specialized demo for Qwen2-VL (Qwen-Image-Edit).

## 🚀 Quick Start

1. Installation
We recommend `uv` for lightning-fast dependency management, but standard `pip` works perfectly too:

```sh
git clone https://github.com/your-username/diffusers-tuner.git
cd diffusers-tuner

# Using uv (Recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

**Requirements**: This repository targets `Python 3.12, Torch 2.10+, and Diffusers 0.37.0+` to support advanced features like QwenImageEdit.

2. Project Structure

A simple yet effective structure helps to understand the core of our framework.

```Plaintext
.
├── configs/             # Hydra configuration center (Adapter, Dataset, Loss, Pipeline, etc.)
├── adapters/            # Lightweight adapter implementations (e.g., LoRA)
├── data/                # Data modules and custom datasets
├── pipelines/           # Specialized pipeline wrappers (e.g., QwenImageEditPlus)
├── tuner.py             # The core tuning loop engine
└── main.py              # Entry point for training
```

## 🐈 How to Use

### 🔥 Launch Tuning

Start your training session with a single command:

```Bash
python main.py
```

Outputs and checkpoints are saved to the outputs directory (customizable in configs/main.yaml).

Take `QwenImageEdit` as example, GPU is 72G in the following situation:

- Resolution: 512 by 512
- Batchsize: 1
- Load text encoder: False
- Num of images: 2
- Precision: bf16
- Gradient accumulation: 1
- Target modules: [attn.to_q, attn.to_k, attn.to_v, attn.to_out.1]


### 🧩 Configuration System
Modify any parameter at runtime via the command line—no code changes required:

```Bash
python main.py \
    data_module.tune_batch_size=4 \   # Change batch size
    adapter.rank=32 \                 # Adjust LoRA rank
    +additional_args="hello_world"    # Inject custom key-value pairs
```

**Module Breakdown**

- Adapter: Defines the fine-tuning mechanism. Our LoRA implementation is cleaner than peft for most use cases, prioritizing simplicity.
- Dataset: Handles complex inputs. Supports images (List of tensors), prompts, targets, and pre-computed prompt_embeds.
- Pipeline: Maps adapters to specific sub-modules (e.g., transformer, unet) within the pretrained pipeline.
- Loss: Configures the optimization objective (e.g., Flow-Matching arguments).

## 📝 Roadmap

- [x] Full Tuning Loop Implementation
- [x] Hierarchical Configuration System
- [x] DDP & Accelerator Support
- [ ] Evaluation Interfaces & Benchmarking
- [ ] Support for more SOTA pipelines