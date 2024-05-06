# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import logging
import os
from contextlib import nullcontext
from pathlib import Path

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SftScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

from causal_generator.datamodule import (
    IGNORE_INDEX,
    DataCollatorForSupervisedDataset,
    GeneratorDataset,
)

from model.vhead_casual import ValueHeadedLLM


tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO
    )


if __name__ == "__main__":
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig))
    # parser.add_argument("--dataset_path", type=str, default="data/leandojo_benchmark_4/random")
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, use_fast=True
    )
    if tokenizer.pad_token is None:
        logging.warning(
            "The tokenizer does not have a pad token, setting it to the eos token."
        )
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # raw_datasets = load_dataset(args.dataset_name)

    # train_dataset = raw_datasets[args.dataset_train_split]
    # eval_dataset = raw_datasets[args.dataset_test_split]

    train_dataset = GeneratorDataset(
        data_path=Path(args.dataset_name) / "train.json",
        corpus=None,
        keep_marks=True,
        preds=None,
        p_drop=0.5,
        normalize_tactics=True,
        max_seq_len=args.max_seq_length,
        tokenizer=tokenizer,
        ignore_index=IGNORE_INDEX,
        is_train=True,
    )

    eval_dataset = GeneratorDataset(
        data_path=Path(args.dataset_name) / "val.json",
        corpus=None,
        keep_marks=True,
        preds=None,
        p_drop=0.5,
        normalize_tactics=True,
        max_seq_len=args.max_seq_length,
        tokenizer=tokenizer,
        ignore_index=IGNORE_INDEX,
        is_train=False,
    )
    collator_fn = DataCollatorForSupervisedDataset(tokenizer)

    ################
    # Optional rich context managers
    ###############
    init_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status("[bold green]Initializing the ValueTrainer...")
    )
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(
            f"[bold green]Training completed! Saving the model to {training_args.output_dir}"
        )
    )

    model = ValueHeadedLLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )  # .to(f"cuda:{local_rank}")

    ################
    # Training
    ################
    with init_context:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator_fn,
            tokenizer=tokenizer,
            # peft_config=get_peft_config(model_config),
            # callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)