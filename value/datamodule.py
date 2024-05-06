from dataclasses import dataclass
import os
import jsonlines
import pickle
import torch
from tqdm import tqdm
from loguru import logger
from typing import Optional, List, Dict, Any, Sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from common import zip_strict


class ValueDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        use_neutral: bool,
        neutral_as_negative: bool,
        max_seq_len: int,
        tokenizer: PreTrainedTokenizer,
    ):
        self.use_neutral = use_neutral
        self.neutral_as_negative = neutral_as_negative
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data_path = data_path

        self._data = self._load_data()

    def _load_data(self):
        data = []
        prompts_to_tokenize = []
        info_list = []
        with jsonlines.open(self.data_path, "r") as reader:
            for obj in reader:
                if obj["value"] == 0.0:
                    # if not use_neutral then only use pos/neg labels to train the value
                    if not self.use_neutral:
                        continue
                    # whether treat neutral as negative
                    elif self.neutral_as_negative:
                        obj["value"] = -1.0
                    else:
                        obj["value"] = 0.0
                prompts_to_tokenize.append(obj["state"])
                info_list.append((obj["state"], obj["value"]))

        token_lengths = self.tokenizer.batch_encode_plus(
            prompts_to_tokenize, return_length=True
        )["length"]

        for tl, (s, v) in zip_strict(token_lengths, info_list):
            if tl <= self.max_seq_len:
                data.append({"state": s, "value": v})
        return data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        state = self._data[index]["state"]
        v = self._data[index]["value"]
        input_ids = self.tokenizer.encode(state, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        value_mask = torch.zeros_like(input_ids, dtype=torch.long)
        value = torch.zeros(len(input_ids), dtype=torch.float)
        value[-1] = v
        value_mask[-1] = 1

        return dict(input_ids=input_ids, labels=value, label_masks=value_mask)


@dataclass
class DataCollatorForValueDataset(object):
    """Collate examples for value fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, label_masks = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "label_masks")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=0.0
        )
        label_masks = torch.nn.utils.rnn.pad_sequence(
            label_masks, batch_first=True, padding_value=0
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            label_masks=label_masks,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
