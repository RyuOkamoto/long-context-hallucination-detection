  # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
  # Licensed under the Apache License, Version 2.0 (the "License").
  # You may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  
  #     http://www.apache.org/licenses/LICENSE-2.0
  
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.

"""
Evaluation Script for Long Context Hallucination Detection

This script provides evaluation capabilities for the long context hallucination detection model.
It loads a trained model and evaluates it on test data, computing various metrics including
F1, Precision, Recall, and ROC-AUC.

Key Features:
- Support for chunked input processing
- Comprehensive evaluation metrics
- Detailed predictions output
- GPU support with proper device handling

Usage:
    python eval.py --model_name_or_path path/to/model \
                   --testing_data_path data/test_all.json \
                   --split --attention_encoder --pad_last --split_inputs

Author: Siyi Liu et al.
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import Dataset
from datetime import timedelta
import csv

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from model import RobertaForSequenceClassificationOurs
from split_chunks import transform_list_of_text_pairs, transform_list_of_text
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument(
        "--explainability",
        action="store_true",
        help="If passed, look at the localization results",
    )

    parser.add_argument(
        "--split_sent",
        action="store_true",
        help="If passed, split response into sentences instead of fixed length chunks",
    )

    parser.add_argument(
        "--add_sep",
        action="store_true",
        help="If passed, add SEP token between context and response chunks.",
    )

    parser.add_argument(
        "--sent_length",
        type=int,
        default=20,
    )


    parser.add_argument(
        "--pad_original",
        action="store_true",
        help="If passed, use original style of padding",
    )

    parser.add_argument(
        "--pair_chunks",
        action="store_true",
        help="If passed, using a NLI style of pairing chunks.",
    )


    parser.add_argument(
        "--pad_last",
        action="store_true",
        help="If passed, pad everything at the end, instead of padding each chunk.",
    )

    parser.add_argument(
        "--split_inputs",
        action="store_true",
        help="If passed, split input into context chunks and response chunks",
    )

    parser.add_argument(
        "--num_chunks1",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--num_chunks2",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=510,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=510,
    )
    parser.add_argument(
        "--minimal_chunk_length",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--maximal_text_length",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="If passed, split input into chunks",
    )

    parser.add_argument(
        "--attention_encoder",
        action="store_true",
        help="If passed, use the attention layer. Else use mean pooler",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--training_data_path",
        type=str,
        help="training data",
        required=True,
    )
    parser.add_argument(
        "--testing_data_path",
        type=str,
        help="testing data",
        required=True,
    )

    parser.add_argument(
        "--backbone_model",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--train_top_k",
        type=int,
        default=1000,
        help="Number of training examples to use from the beginning of the training dataset.",
    )
    args = parser.parse_args()

    # Sanity checks

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def preprocess_function(examples):
    # Tokenize the texts
    if "roberta" in args.model_name_or_path:
        if args.split:
            if args.split_inputs:
                # print("Splitting inputs contexts response!")
                batch = transform_list_of_text_pairs(examples["context"],examples["response"], tokenizer, args.chunk_size, args.stride,
                                                args.minimal_chunk_length, args.num_chunks1, args.num_chunks2, args.pad_last, args.pad_original, args.maximal_text_length,args.split_sent, args.sent_length, args.pair_chunks)

            else:
                batch = transform_list_of_text(examples["text"], tokenizer, args.chunk_size, args.stride,
                                                args.minimal_chunk_length, args.num_chunks1, args.pad_last, args.pad_original, args.maximal_text_length)


        else:
            batch = tokenizer(
                examples["text"],
                padding=padding,
                max_length=args.max_length,
                truncation=True,
            )

        batch['labels'] = examples['labels']

        return batch

def main():
    args = parse_args()
    with open(args.testing_data_path, "r") as f:
        test = json.load(f)
    test_dataset = Dataset.from_list(test)

    num_labels = 2
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    config.pad_token_id = tokenizer.pad_token_id
    config.problem_type = "single_label_classification"
    config.split = args.split
    config.attention_encoder = args.attention_encoder
    config.pad_last = args.pad_last
    config.explainability = args.explainability
    config.num_chunks_context = args.num_chunks1
    config.split_inputs = args.split_inputs
    config.add_sep = args.add_sep
    config.split_sent = args.split_sent
    config.pair_chunks = args.pair_chunks

    if args.backbone_model:

        model = RobertaForSequenceClassificationOurs.from_pretrained(
            args.backbone_model,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = RobertaForSequenceClassificationOurs.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            trust_remote_code=args.trust_remote_code,
        )

    model.to(args.device)
    model.eval()

    padding = "max_length" if args.pad_to_max_length else False


    eval_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Running tokenizer on test dataset",
    )

    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    metric1 = evaluate.load("f1")
    metric2 = evaluate.load("precision")
    metric3 = evaluate.load("recall")
    metric4 = evaluate.load("accuracy")
    metric5 = evaluate.load("roc_auc")

    preds = []
    refs = []
    txt = []
    txt2 = []
    all_scores = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references, logits, input_ids, response_input_ids = predictions, batch["labels"], outputs.logits, batch["input_ids"], batch["response_input_ids"]

        scores = logits.softmax(dim=1)
        scores = [scores[ind][1] for ind in range(len(scores))]

        new_preds = []
        for each in scores:
            if each > 0.2:
                new_preds.append(1)
            else:
                new_preds.append(0)
        predictions = new_preds

        metric1.add_batch(
            predictions=predictions,
            references=references,
        )
        metric2.add_batch(
            predictions=predictions,
            references=references,
        )
        metric3.add_batch(
            predictions=predictions,
            references=references,
        )
        metric5.add_batch(
            prediction_scores=scores,
            references=references,
        )

        for ind in range(len(predictions)):
            preds.append(int(predictions[ind]))
            refs.append(int(references[ind].detach().cpu().numpy()))
            txt.append(tokenizer.decode(list(input_ids[ind].reshape(-1).detach().cpu().numpy())))
            txt2.append(tokenizer.decode(list(response_input_ids[ind].reshape(-1).detach().cpu().numpy())))
            all_scores.append(scores[ind])

    eval_metric1 = metric1.compute()
    eval_metric2 = metric2.compute()
    eval_metric3 = metric3.compute()
    eval_metric5 = metric5.compute()
    print(f" f1: {eval_metric1}")
    print(f" precision: {eval_metric2}")
    print(f" recall: {eval_metric3}")
    print(f" ROCAUC: {eval_metric5}")


if __name__ == "__main__":
    main()