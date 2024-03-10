# export CUDA_VISIBLE_DEVICES=7

python mcts_prover/evaluate.py \
    --data-path data/leandojo_benchmark/random/ \
    --ckpt_path /data/workspace/muning/GloveInDark/models/leandojo-lean4-tacgen-byt5-small \
    --split test \
    --num-cpus 32 \
    --with-gpus \
    | tee run_20240310.log