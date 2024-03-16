# MCTS Prover in Leandojo

## Installation
see `original_README` or the [online version](https://github.com/lean-dojo/ReProver?tab=readme-ov-file#requirements)

### Data Preparation
see `original_README` or the [online version](https://github.com/lean-dojo/ReProver?tab=readme-ov-file#requirements)

### Download Model
Just clone the tac_gen model
```
git clone https://huggingface.co/kaiyuy/leandojo-lean4-tacgen-byt5-small
```


## Run
modify `--data-path` and `--ckpt-path` in `run_tp.sh`, the maximal value of `num-cpus` are limited to `gpu_number * 4`.
```
export PYTHONPATH=$(pwd)
sh run_tp.sh

# original best-first-search
sh run_tp_bfs.sh
```

## important env variables
```
export TMP_DIR= {where to create tmp files}
export CACHE_DIR= {where you trace_repos}


export TACTIC_CPU_LIMIT= {#threads when launch lean4 services}
export RAY_NUM_GPU_PER_WORKER= {gpu resources for each worker}
```