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
```