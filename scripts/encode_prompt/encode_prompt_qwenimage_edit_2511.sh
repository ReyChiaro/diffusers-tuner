export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

MODEL_PATH="Qwen/Qwen-Image-Edit-2511"
DATA_PATH="dataset/dummy/dummy.json"
ENCODED_PROMPT_DIR="dataset/dummy/prompt_embeds"

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --gpu_ids 7 \
    --mixed_precision no \
    --dynamo_backend no \
    encode_prompt.py --config-name encode_prompt \
    tune.project_name=encode-propmt-qwenimage-2511 \
    dataset=coco \
    dataset.data_file=${DATA_PATH} \
    pipeline=qwenimage_edit_plus \
    pipeline.pretrained_model_name_or_path=${MODEL_PATH} \
    "pipeline.load_modules=[processor,tokenizer,text_encoder]" \
    +prompt_embeds_save_dir=${ENCODED_PROMPT_DIR} \

