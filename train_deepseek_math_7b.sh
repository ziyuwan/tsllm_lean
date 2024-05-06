export CUDA_VISIBLE_DEVICES=5,6

torchrun --nproc_per_node=2 --master-port=29423 causal_generator/main.py \
    --model_name_or_path="/data/workspace/muning/GloveInDark/MATH_PJ//models/deepseek-math-7b-base/" \
    --dataset_name="data/leandojo_benchmark_4/random" \
    --report_to="tensorboard" \
    --learning_rate=2e-5 \
    --warmup_ratio=0.03 \
    --lr_scheduler_type="cosine" \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=8 \
    --output_dir="sft_leandojo_deepseek_math_7b" \
    --torch_dtype=bfloat16 \
    --max_seq_length 512 \
    --attn_implementation=flash_attention_2 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --save_strategy "epoch" \
    --save_only_model True \
    --bf16 \
    --gradient_checkpointing