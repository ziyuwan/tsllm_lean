"""Lightning module for the tactic generator."""

import requests
import torch
from generator.model import TacticGenerator
from lean_dojo import Pos
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple

from common import (
    zip_strict,
    remove_marks,
    IndexedCorpus,
    get_optimizers,
    load_checkpoint,
    format_augmented_state,
)
from retrieval.model import PremiseRetriever
from dataclasses import dataclass, asdict
from functools import partial

from vllm import SamplingParams

torch.set_float32_matmul_precision("medium")

def vllm_api_call(
    query_str: str,
    worker_address="http://0.0.0.0:8000",
    **sampling_params,
):
    gen_params = {
        "prompt": query_str,
        "stream": False,
        **sampling_params
    }
    headers = {"User-Agent": "LeanDojo Prover Client"}
    response = requests.post(
        worker_address + "/generate",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    results = response.json()
    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"]
    # avg_len_logps = [clp / otl for clp, otl in zip(cum_logps, output_token_lens)]
    return results["text"], cum_logps


@dataclass
class GeneratorConfig:
    num_beams: int
    max_inp_seq_len: int
    max_oup_seq_len: int
    length_penalty: float

    def dict(self):
        return asdict(self)

class SimpleRetrievalAugmentedGenerator(TacticGenerator):
    def __init__(
        self,
        model_name: str,
        num_beams: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        length_penalty: float = 0.0,
        ret_ckpt_path: Optional[str] = None,
        device="cpu",
    ) -> None:
        super().__init__()
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len

        self.device = torch.device(device)

        if ret_ckpt_path is None:
            logger.info("Without retrieval")
            self.retriever = None
        else:
            raise NotImplementedError

        assert num_beams == 1

        self.llm_gen_fn = partial(
            vllm_api_call, 
            temperature=1.0,
            top_k=100,
            max_tokens=max_oup_seq_len,
        )

    ##############
    # Prediction #
    ##############

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raw_output_text, raw_scores = self.llm_gen_fn(state+"\n", n=num_samples)
        output_text = []
        output_score = []
        for j in range(len(raw_output_text)):
                t = remove_marks(raw_output_text[j])
                if len(t) > 0 and t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])
        
        return list(zip_strict(output_text, output_score))
    
    def batch_generate(self, state: List[str], file_path: List[str], theorem_full_name: List[str], theorem_pos: List[Pos], num_samples: int) -> List[List[Tuple[str | float]]]:
        raise NotImplementedError


if __name__ == "__main__":
    gen = SimpleRetrievalAugmentedGenerator(
        None, 1, 1024, 64
    ) 

    print(gen.generate("Who are you?\n", None, None, None, 1))