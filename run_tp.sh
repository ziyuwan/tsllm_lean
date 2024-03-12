set -e 
# export CUDA_VISIBLE_DEVICES=7
export TMP_DIR=/data/workspace/muning/GloveInDark/ReProver/tmpdir
export CACHE_DIR=/data/workspace/muning/GloveInDark/ReProver/.cache/lean_dojo

python mcts_prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --ckpt_path /data/workspace/muning/GloveInDark/models/leandojo-lean4-tacgen-byt5-small \
    --split test \
    --num-cpus 64 \
    --with-gpus \
    | tee run_20240311.log