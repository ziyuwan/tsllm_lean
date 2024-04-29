"""Data module for the tactic generator."""

from dataclasses import dataclass
import os
import json
import pickle
import torch
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any, Sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from common import (
    Batch,
    Corpus,
    Example,
    format_state,
    remove_marks,
    format_tactic,
    format_augmented_state,
)

IGNORE_INDEX = -100


class GeneratorDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        corpus: Corpus,
        keep_marks: bool,
        preds: List[Dict[str, Any]],
        p_drop: float,
        max_seq_len: int,
        normalize_tactics: bool,
        tokenizer: PreTrainedTokenizer,
        is_train: bool,
        ignore_index: int,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.keep_marks = keep_marks
        self.preds = preds
        self.p_drop = p_drop
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.ignore_index = ignore_index
        self.data = self._load_data(data_path, normalize_tactics)

    def _load_data(self, data_path: str, normalize_tactics: bool) -> List[Example]:
        data = []
        for thm in tqdm(json.load(open(data_path))):
            for tac in thm["traced_tactics"]:
                if "annotated_tactic" in tac:
                    tactic = format_tactic(*tac["annotated_tactic"], normalize_tactics)
                else:
                    tactic = format_tactic(tac["tactic"], [], normalize_tactics)
                if not self.keep_marks:
                    tactic = remove_marks(tactic)
                # FIXME(ziyu): very slow !
                if (
                    len(
                        self.tokenizer.encode(
                            format_state(tac["state_before"])
                            + "\n"
                            + tactic
                            + self.tokenizer.eos_token
                        )
                    )
                    > self.max_seq_len
                ):
                    continue
                data.append(
                    {
                        "url": thm["url"],
                        "commit": thm["commit"],
                        "file_path": thm["file_path"],
                        "full_name": thm["full_name"],
                        "state": format_state(tac["state_before"]),
                        "tactic": tactic,
                    }
                )

        logger.info(f"{len(data)} examples loaded")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]

        if self.preds is not None:
            file_path = ex["file_path"]
            pred = self.preds[(file_path, ex["full_name"], ex["state"])]
            ex["state"] = format_augmented_state(
                ex["state"],
                pred["retrieved_premises"],
                self.max_seq_len,
                self.p_drop if self.is_train else 0.0,
            )

        if not self.keep_marks:
            ex["state"] = remove_marks(ex["state"])

        query_ids = self.tokenizer.encode(ex["state"] + "\n", add_special_tokens=True)
        len_q = len(query_ids)
        input_ids = self.tokenizer.encode(
            ex["state"] + "\n" + ex["tactic"] + self.tokenizer.eos_token,
            add_special_tokens=True,
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(input_ids, dtype=torch.long)
        labels[:len_q] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
