export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

MODEL_PATH="Qwen/Qwen-Image-Edit-2511"
DATA_PATH="dataset/dummy/dummy_bucket.json"

accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    --gpu_ids 3,4,6,7 \
    --multi_gpu \
    --mixed_precision bf16 \
    --dynamo_backend no \
    tune.py --config-name tune \
    tune.project_name=tune-qwenimage-2511 \
    tune.max_tuning_steps=10000 \
    tune.save_steps=1000 \
    tune.eval_steps=1000 \
    tune.mixed_precision=bf16 \
    pipeline=qwenimage_edit_plus \
    pipeline.pretrained_model_name_or_path=${MODEL_PATH} \
    "pipeline.load_modules=[scheduler,processor,tokenizer,text_encoder,vae,transformer]" \
    "pipeline.tune_modules=[transformer]" \
    adapter=lora \
    adapter.rank=256 \
    "adapter.target_modules=[attn.to_q,attn.to_k,attn.to_v,attn.to_out,attn.add_q_proj,attn.add_k_proj,attn.add_v_proj,attn.to_add_out,img_mlp.net.2,txt_mlp.net.2]" \
    data.tune_batch_size=1 \
    dataset=coco \
    dataset.bucket_dataset=ture \
    dataset.data_file=${DATA_PATH} \
