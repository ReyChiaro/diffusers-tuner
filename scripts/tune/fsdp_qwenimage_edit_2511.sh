export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

MODEL_PATH="/mtc/models/Qwen-Image-Edit-2511"
DATA_PATH="dataset/coco/coco_bucket.json"

accelerate launch \
    --num_machines 1 \
    --num_processes 2 \
    --gpu_ids 6,7 \
    --use_fsdp \
    --mixed_precision bf16 \
    --dynamo_backend no \
    tune.py --config-name tune \
    tune.project_name=full-tune-qwenimage-2511 \
    tune.max_tuning_steps=10 \
    tune.save_steps=5 \
    tune.eval_steps=5 \
    tune.mixed_precision=bf16 \
    pipeline=qwenimage_edit_plus \
    pipeline.pretrained_model_name_or_path=${MODEL_PATH} \
    pipeline.load_modules=[scheduler,processor,tokenizer,text_encoder,vae,transformer] \
    pipeline.tune_modules=[transformer] \
    "pipeline.full_tune_modules={transformer:[transformer_blocks.59]}" \
    data.batch_size=4 \
    dataset=coco \
    dataset.bucket_dataset=ture \
    dataset.data_file=${DATA_PATH} \
