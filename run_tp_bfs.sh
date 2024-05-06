set -e 
export CUDA_VISIBLE_DEVICES=1,2
export TMP_DIR=/data/workspace/muning/GloveInDark/MATH_PJ/ReProver/tmpdir
export CACHE_DIR=/data/workspace/muning/GloveInDark/MATH_PJ/ReProver/test_cache/lean_dojo

export TACTIC_CPU_LIMIT=1
export RAY_NUM_GPU_PER_WORKER=1

    # --ckpt_path /data/workspace/muning/GloveInDark/MATH_PJ/models/leandojo-lean4-tacgen-byt5-small \
python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --ckpt_path "checkpoint-1463" \
    --split test \
    --num-cpus 4
    # --with-gpus \