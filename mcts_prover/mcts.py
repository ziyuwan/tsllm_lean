import math
from pathlib import Path
import sys
import time
import numpy as np
import ray
import torch
from lean_dojo.interaction.dojo import TacticState
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    LeanError,
    TimeoutError,
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
    DojoHardTimeoutError,
)
from common import zip_strict
from mcts_prover.mcts_node import (
    Node,
    ProofFinishedNode,
    Status,
    Edge,
    ProofFinished,
    ErrorNode,
    InternalTreeNode,
)
from mcts_prover.save_tree import _node_to_dict, _find_tree_root
from mcts_prover.ucb import compute_puct
from generator.model import FixedTacticGenerator
from generator.simplified_model import (
    SimpleRetrievalAugmentedGenerator,
    GeneratorConfig,
)
from dataclasses import dataclass
from typing import Optional, List, Tuple, Iterable, Union
from prover.proof_search import SearchResult
from loguru import logger
from ray.util.actor_pool import ActorPool
import json

import os

RAY_NUM_GPU_PER_WORKER = float(os.environ.get("RAY_NUM_GPU_PER_WORKER", 1))


@dataclass
class MCTSConfig:
    pb_c_init: float
    pb_c_base: float

    ## mcts.rollout do not use this random noise
    # root_dirichlet_alpha: float
    # root_noise_weight: float


class MCTSProver:
    def __init__(
        self,
        tac_gen,
        timeout: int,
        num_sampled_tactics: int,
        mcts_config: MCTSConfig,
        save_tree_dir: Optional[str] = None,
        debug=bool,
    ):

        self.tac_gen = tac_gen
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

        self.config = mcts_config
        if save_tree_dir is not None:
            self._save_tree_dir = Path(save_tree_dir)
            assert self._save_tree_dir.exists()
        else:
            self._save_tree_dir = None

    def add_logger(self, logger_path: str):
        print("Logging to {}".format(logger_path))
        assert Path(logger_path).parent.exists()
        logger.add(logger_path, enqueue=True, level="INFO")

    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:

        if self._save_tree_dir is not None:
            save_path0 = self._save_tree_dir / f"{thm.uid}.json"
            save_path1 = self._save_tree_dir / f"{thm.uid1}.json"

            if save_path0.exists() or save_path1.exists():
                save_path = save_path0 if save_path0.exists() else save_path1
            
            if save_path.exists():
                logger.info(
                    f"Search tree already exists. Loading results from {save_path}"
                )
                with open(save_path, "r") as f:
                    result_ckpt = json.load(f)
                    result = SearchResult(theorem=thm, **result_ckpt["result"])
                    return result

        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.posision = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0
        self.num_total_nodes = 0

        if isinstance(self.tac_gen, FixedTacticGenerator):
            imps = [self.tac_gen.module]
        else:
            imps = []

        try:
            with Dojo(thm, hard_timeout=60 + self.timeout, additional_imports=imps) as (
                dojo,
                init_state,
            ):
                self.dojo = dojo
                self.root = InternalTreeNode(state=init_state, critic_value=0.0)
                with torch.no_grad():
                    try:
                        self._alphazero_search()
                    except DojoCrashError:
                        logger.warning(f"Dojo crashed when proving {thm}")
                        pass

            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=self.num_total_nodes,
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)
            if self._save_tree_dir is not None:
                assert self.root.is_root
                # save_path = self._save_tree_dir / f"{thm.uid}.json"
                save_path = self._save_tree_dir / f"{thm.uid1}.json"
                logger.info(f"Saving search tree to {save_path}")
                search_tree = _node_to_dict(self.root)
                res_json = {
                    "status": str(result.status),
                    "proof": result.proof,
                    "actor_time": result.actor_time,
                    "environment_time": result.environment_time,
                    "total_time": result.total_time,
                    "num_total_nodes": result.num_total_nodes,
                    "num_searched_nodes": result.num_searched_nodes,
                }
                json2save = {"search_tree": search_tree, "result": res_json}
                json.dump(json2save, open(save_path, "w"))

            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    def _alphazero_search(self):
        time_start = time.monotonic()

        while True:
            try:
                self._simulation(self.root)
            except DojoHardTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                self.root.status = Status.OPEN
                logger.info("Search timed out.")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break

    def _simulation(self, node: InternalTreeNode):
        """
        select from given node untill a leaf node and then
        expand the leaf value if possible
        then evaluate the leaf value
        then backup
        """
        while not node.is_leaf:
            tactic, node = self._select_child(node)
            logger.debug(f"Select tactic: {tactic}")

        if not node.is_terminal:
            if isinstance(node.state, TacticState):
                ts = node.state.pp
            else:
                ts = node.state.unsolved_tactic_state
            suggestions = self._generate_tactics(ts)
            results = [
                self._run_tactic(node, tactic, logprob)
                for tactic, logprob in suggestions
            ]
            node.out_edges = results
            # TODO(ziyu): compute_value here or in _run_tactics or in _generate_tactics,
            #  For me I think here is the most flexible

            self.num_expansions += 1
            self.num_total_nodes += len(results)

        node.backup(node.value)

    def _select_child(self, node: InternalTreeNode):
        assert len(node.out_edges) > 0, node

        scores: np.ndarray = compute_puct(
            parent_node=node,
            pb_c_base=self.config.pb_c_base,
            pb_c_init=self.config.pb_c_init,
        )

        best_i = np.argmax(scores)
        action = node.out_edges[best_i].tactic
        child = node.out_edges[best_i].dst

        return action, child

    def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        path = str(self.theorem.file_path)

        if self.theorem.repo != self.repo:
            path = self.theorem.repo.get_packages_dir() / self.theorem.repo.name / path

        suggestions = self.tac_gen.generate(
            state=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.posision,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    # (ziyu): like env step in RL
    def _run_tactic(self, node: InternalTreeNode, tactic: str, logprob: float) -> Edge:
        assert node.is_leaf

        t0 = time.monotonic()
        response = self.dojo.run_tac(node.state, tactic)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        # Build a new node
        if isinstance(response, ProofFinished):
            result_node = ProofFinishedNode(response)
        elif type(response) in (
            LeanError,
            TimeoutError,
            ProofGivenUp,
        ):
            result_node = ErrorNode(response)
        else:
            assert isinstance(response, TacticState)
            result_node = InternalTreeNode(
                state=response,
                critic_value=0.0,
                # critic_value=node.cum_logp + logprob,
                cum_logp=node.cum_logp + logprob,
            )

        edge = Edge(tactic=tactic, logp=logprob, src=node, dst=result_node)
        result_node.in_edge = edge
        return edge


@ray.remote
class CpuProver(MCTSProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a CPU."""

    def __init__(
        self,
        model_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        generator_config: GeneratorConfig,
        mcts_config: MCTSConfig,
        save_tree_dir: Optional[str],
        debug: bool,
    ) -> None:
        if model_path is None:
            tac_gen = FixedTacticGenerator(tactic, module)
        else:
            tac_gen = SimpleRetrievalAugmentedGenerator(
                model_path, device=torch.device("cpu"), **generator_config.dict()
            )
            if tac_gen.retriever is not None:
                if indexed_corpus_path is not None:
                    tac_gen.retriever.load_corpus(indexed_corpus_path)
                tac_gen.retriever.reindex_corpus(batch_size=32)
        super().__init__(
            tac_gen,
            timeout,
            num_sampled_tactics,
            mcts_config,
            save_tree_dir,
            debug,
        )


@ray.remote(num_gpus=RAY_NUM_GPU_PER_WORKER)
class GpuProver(MCTSProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a GPU."""

    def __init__(
        self,
        model_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        generator_config: GeneratorConfig,
        mcts_config: MCTSConfig,
        save_tree_dir: Optional[str],
        debug: bool,
    ) -> None:
        if model_path is None:
            tac_gen = FixedTacticGenerator(tactic, module)
        else:
            # tac_gen = RetrievalAugmentedGenerator.load(
            #     ckpt_path, device=torch.device("cuda"), freeze=True
            # )
            tac_gen = SimpleRetrievalAugmentedGenerator(
                model_path, device=torch.device("cuda"), **generator_config.dict()
            )
            if tac_gen.retriever is not None:
                if indexed_corpus_path is not None:
                    tac_gen.retriever.load_corpus(indexed_corpus_path)
                tac_gen.retriever.reindex_corpus(batch_size=32)
        super().__init__(
            tac_gen,
            timeout,
            num_sampled_tactics,
            mcts_config,
            save_tree_dir,
            debug,
        )


class DistributedProver:
    """A distributed prover that uses Ray to parallelize the proof search.

    It is a wrapper around `CpuProver` and `GpuProver` that handles the different
    devices and different number of concurrent provers.
    """

    def __init__(
        self,
        model_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        num_cpus: int,
        with_gpus: bool,
        timeout: int,
        num_sampled_tactics: int,
        generator_config: GeneratorConfig,
        mcts_config: MCTSConfig,
        logger_path: Optional[str] = None,
        save_tree_dir: Optional[str] = None,
        debug: Optional[bool] = False,
    ) -> None:
        if model_path is None:
            assert tactic and not indexed_corpus_path
        else:
            assert not tactic and not module
        self.distributed = num_cpus > 1
        self._logger_path = logger_path

        if not self.distributed:
            if model_path is None:
                tac_gen = FixedTacticGenerator(tactic, module)
            else:
                device = torch.device("cuda") if with_gpus else torch.device("cpu")
                tac_gen = SimpleRetrievalAugmentedGenerator(
                    model_path, device=device, **generator_config.dict()
                )
                if tac_gen.retriever is not None:
                    assert indexed_corpus_path is not None
                    tac_gen.retriever.load_corpus(indexed_corpus_path)
            self.prover = MCTSProver(
                tac_gen, timeout, num_sampled_tactics, mcts_config, save_tree_dir, debug
            )
            return

        ray.init()
        if with_gpus:
            logger.info(f"Launching {num_cpus} GPU workers.")
            provers = [
                GpuProver.remote(
                    model_path,
                    indexed_corpus_path,
                    tactic,
                    module,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    generator_config=generator_config,
                    mcts_config=mcts_config,
                    save_tree_dir=save_tree_dir,
                    debug=debug,
                )
                for _ in range(num_cpus)
            ]
        else:
            logger.info(f"Launching {num_cpus} CPU workers.")
            provers = [
                CpuProver.remote(
                    model_path,
                    indexed_corpus_path,
                    tactic,
                    module,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    generator_config=generator_config,
                    mcts_config=mcts_config,
                    save_tree_dir=save_tree_dir,
                    debug=debug,
                )
                for _ in range(num_cpus)
            ]

        # XXX(ziyu): make sure the logger path can be access on each node
        if logger_path is not None:
            for prover in provers:
                prover.add_logger.remote(logger_path)

        self.prover_pool = ActorPool(provers)

    def search_unordered(
        self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]
    ) -> List[SearchResult]:
        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
        if not self.distributed:
            return [
                self.prover.search(repo, thm, pos)
                for thm, pos in zip_strict(theorems, positions)
            ]

        try:
            results = list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(repo, x[0], x[1]),
                    zip_strict(theorems, positions),
                )
            )
        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)

        return results
