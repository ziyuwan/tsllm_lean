# MCTS Prover in Leandojo

## Installation
see `original_README` or the [online version](https://github.com/lean-dojo/ReProver?tab=readme-ov-file#requirements)

### Offline Running Version
Install my fork of pure offline mode
```
git clone git@github.com:ziyuwan/ZYLeanDojo.git
cd ZYLeanDojo
git checkout offline_mode

pip install -e .
```
After `trace_repos.py`

```
# export CACHE_DIR={your cache dir}
# export TMP_DIR={your tmp dir}

PURE_OFFLINE_MODE=0 python get_offline_cache.py --data-path {the data path} --split test --num-build-workers {NUM_WORKER}
```
It will first save a cached pickle for each repos and theorems. Then it will try to build each theorem with `num-build-workers` to download all dependencies.


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

## Data Sampling
We support tree-structured data storage for MCTS search.
To sample data from training set. First copy the cached_repo.pickle file
```
cp repos_to_cp/random/cached_repo_train.pickle data/leandojo_benchmark_4/random
```

Then in the `run_tp.sh` script, add new arguments for
- Tree data dir: `--save-tree-dir`, this should be a path **every worker** can access
- theorem filtering by number of steps: `--min-num-steps` and `--max-num-steps`, for example `--min-num-steps 2 --max-num-steps 10`, which will get 27k problem from the original 97k training set.

Then running the mcts search script.

## important env variables
```
export TMP_DIR= {where to create tmp files}
export CACHE_DIR= {where you trace_repos}


export TACTIC_CPU_LIMIT= {#threads when launch lean4 services}
export RAY_NUM_GPU_PER_WORKER= {gpu resources for each worker}
```

