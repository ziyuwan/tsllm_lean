"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""

import os
import uuid
import json
import pickle
import hashlib
import argparse
from loguru import logger
from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache
from causal_generator.simplified_model import GeneratorConfig
from datetime import datetime

from common import set_logger
from prover.proof_search import Status, DistributedProver
from lean_dojo.data_extraction.lean import PURE_OFFLINE_MODE
from pathlib import Path
from transformers import AutoConfig, PretrainedConfig


def _get_theorems(
    data_path: str,
    split: str,
    file_path: str,
    full_name: str,
    name_filter: str,
    num_theorems: int,
    min_num_steps: Optional[int] = None,
    max_num_steps: Optional[int] = None,
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    repo, theorems, positions = _get_theorems_from_files(
        data_path,
        split,
        file_path,
        full_name,
        name_filter,
        num_theorems,
        min_num_steps,
        max_num_steps,
    )

    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        assert is_available_in_cache(
            r
        ), f"{r} has not been traced yet. Please use LeanDojo to trace it so that it's available in the cache."

    return repo, theorems, positions


def _get_theorems_from_files(
    data_path: str,
    split: str,
    file_path: Optional[str],
    full_name: Optional[str],
    name_filter: Optional[str],
    num_theorems: Optional[int],
    min_num_steps: Optional[int] = None,
    max_num_steps: Optional[int] = None,
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:

    if not PURE_OFFLINE_MODE:
        data = json.load(open(os.path.join(data_path, f"{split}.json")))
        theorems = []
        positions = []

        for t in data:
            repo = LeanGitRepo(t["url"], t["commit"])
            theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
            positions.append(Pos(*t["start"]))

        # theorems = sorted(
        #     theorems,
        #     key=lambda t: hashlib.md5(
        #         (str(t.file_path) + ":" + t.full_name).encode()
        #     ).hexdigest(),
        # )
        metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
        repo = LeanGitRepo(
            metadata["from_repo"]["url"], metadata["from_repo"]["commit"]
        )
        cached_git_data = {
            "meta_repo": repo,
            "theorems": theorems,
            "positions": positions,
        }
        pickle.dump(
            cached_git_data, open(Path(data_path) / f"cached_repo_{split}.pickle", "wb")
        )
        print("Dump to ", Path(data_path) / f"cached_repo_{split}.pickle")
    else:
        cached_git_data = pickle.load(
            open(Path(data_path) / f"cached_repo_{split}.pickle", "rb")
        )
        repo = cached_git_data["meta_repo"]
        theorems = cached_git_data["theorems"]
        positions = cached_git_data["positions"]

        def _test_repo(r):
            r.get_packages_dir()
            r.get_build_dir()

        _test_repo(repo)
        for thm in theorems:
            _test_repo(thm.repo)

    if min_num_steps is not None or max_num_steps is not None:
        logger.info(
            "Filtering theorems by number of steps: min={}, max={}".format(
                min_num_steps, max_num_steps
            )
        )

    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    selected_theorems = []
    selected_positions = []
    for idx, t in enumerate(data):
        if file_path is not None and t["file_path"] != file_path:
            continue
        if full_name is not None and t["full_name"] != full_name:
            continue
        if name_filter is not None and not hashlib.md5(
            t["full_name"].encode()
        ).hexdigest().startswith(name_filter):
            continue
        if min_num_steps is not None and len(t["traced_tactics"]) < min_num_steps:
            continue
        if max_num_steps is not None and len(t["traced_tactics"]) > max_num_steps:
            continue
        assert theorems[idx].full_name == t["full_name"]
        selected_theorems.append(theorems[idx])
        selected_positions.append(positions[idx])

    if num_theorems is not None:
        selected_theorems = selected_theorems[:num_theorems]
        selected_positions = selected_positions[:num_theorems]

    logger.info(f"{len(selected_theorems)} theorems loaded from {data_path}")
    return repo, selected_theorems, selected_positions


def evaluate(
    data_path: str,
    exp_id: Optional[str] = None,
    split: str = "val",
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    indexed_corpus_path: Optional[str] = None,
    tactic: Optional[str] = None,
    module: Optional[str] = None,
    num_sampled_tactics: int = 64,
    num_beams: int = 4,
    max_inp_seq_len: int = 1024,
    max_oup_seq_len: int = 1024,
    length_penalty: float = 0.0,
    timeout: int = 600,
    num_cpus: int = 1,
    with_gpus: bool = False,
    verbose: bool = False,
) -> float:
    set_logger(verbose)
    logger_path = Path(f"bfs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logger_path = logger_path.absolute().as_posix()
    logger.add(logger_path, enqueue=True)

    repo, theorems, positions = _get_theorems(
        data_path, split, file_path, full_name, name_filter, num_theorems
    )

    generator_config = GeneratorConfig(
        num_beams=num_beams,
        max_inp_seq_len=max_inp_seq_len,
        max_oup_seq_len=max_oup_seq_len,
        length_penalty=length_penalty,
    )
    # Search for proofs using multiple concurrent provers.
    ckpt_cfg: PretrainedConfig = AutoConfig.from_pretrained(ckpt_path)
    prover = DistributedProver(
        ckpt_path,
        ckpt_cfg.is_encoder_decoder,
        indexed_corpus_path,
        tactic,
        module,
        num_cpus,
        with_gpus=with_gpus,
        timeout=timeout,
        num_sampled_tactics=num_sampled_tactics,
        generator_config=generator_config,
        logger_path=logger_path,
        debug=verbose,
    )
    results = prover.search_unordered(repo, theorems, positions)

    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if num_proved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_proved / (num_proved + num_failed)

    # Save the results.
    if exp_id is None:
        exp_id = str(uuid.uuid4())
    pickle_path = f"{exp_id}_results.pickle"
    pickle.dump(results, open(pickle_path, "wb"))
    logger.info(f"Results saved to {pickle_path}")

    return pass_1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument("--tactic", type=str, help="The tactic to evaluate.")
    parser.add_argument("--module", type=str, help="The module to import the tactic.")
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-inp-seq-len",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--max-oup-seq-len",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--num-cpus", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--with-gpus", action="store_true", help="Use GPUs for proof search."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    args = parser.parse_args()

    assert args.ckpt_path or args.tactic

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    pass_1 = evaluate(
        args.data_path,
        args.exp_id,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
        args.ckpt_path,
        args.indexed_corpus_path,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.num_beams,
        args.max_inp_seq_len,
        args.max_oup_seq_len,
        args.length_penalty,
        args.timeout,
        args.num_cpus,
        args.with_gpus,
        args.verbose,
    )

    logger.info(f"Pass@1: {pass_1}")


if __name__ == "__main__":
    main()
