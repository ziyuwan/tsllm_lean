"""Lightning module for the tactic generator."""

import torch
from generator.model import TacticGenerator, TopkAccuracy
import pickle
from lean_dojo import Pos
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple
from transformers import T5ForConditionalGeneration, AutoTokenizer

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


torch.set_float32_matmul_precision("medium")


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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

        # self.topk_accuracies = dict()
        # for k in range(1, num_beams + 1):
        #     acc = TopkAccuracy(k)
        #     self.topk_accuracies[k] = acc
        #     self.add_module(f"top{k}_acc_val", acc)

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
        return self.batch_generate(
            [state], [file_path], [theorem_full_name], [theorem_pos], num_samples
        )[0]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        logger.debug(state)
        if self.retriever is not None:
            retrieved_premises, _ = self.retriever.retrieve(
                state,
                file_path,
                theorem_full_name,
                theorem_pos,
                self.eval_num_retrieved,
            )
            state = [
                format_augmented_state(s, premises, self.max_inp_seq_len, p_drop=0.0)
                for s, premises in zip_strict(state, retrieved_premises)
            ]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_inp_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_oup_seq_len,
            num_beams=num_samples,
            length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        raw_scores = output.sequences_scores.tolist()
        tactics_with_scores = []

        for i in range(len(state)):
            output_text = []
            output_score = []

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = remove_marks(raw_output_text[j])
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores
