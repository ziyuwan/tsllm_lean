from transformers import AutoTokenizer
from value.model.vhead_casual import ValueHeadedLLM
import torch
from value.datamodule import ValueDataset, DataCollatorForValueDataset
from torch.utils.data import DataLoader

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.float32,
    use_cache=False,
    device_map=None,
)


model = ValueHeadedLLM.from_pretrained(
    "../models/deepseek-math-7b-base/", **model_kwargs
)

tok = AutoTokenizer.from_pretrained("../models/deepseek-math-7b-base/")
tok.pad_token = tok.eos_token

v_ds = ValueDataset(
    data_path="tmp_v.jsonl",
    use_neutral=True,
    neutral_as_negative=False,
    max_seq_len=512,
    tokenizer=tok,
)

dataloader = DataLoader(
    v_ds, batch_size=2, shuffle=True, collate_fn=DataCollatorForValueDataset(tok)
)

batch = next(iter(dataloader))
batch["labels"] = torch.arange(batch["input_ids"].size(1)).unsqueeze(0).repeat(2, 1)

output = model(**batch)
