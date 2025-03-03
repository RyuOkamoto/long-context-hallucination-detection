# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Finetuning a 🤗 Transformers model for sequence classification on GLUE."""

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


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.43.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")



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
    args = parser.parse_args()

    # Sanity checks

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args



def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    timeout = [InitProcessGroupKwargs(timeout=timedelta(seconds=1800))]
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir, kwargs_handlers=timeout) if args.with_tracking else Accelerator(kwargs_handlers=timeout)
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
        # Loading the dataset from local csv or json file.
    # data_files = {}
    # if args.train_file is not None:
    #     data_files["train"] = args.train_file
    # if args.validation_file is not None:
    #     data_files["validation"] = args.validation_file
    # extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
    # raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    with open(args.testing_data_path, "r") as file:
        test = json.load(file)

    with open(args.training_data_path, "r") as file:
        train = json.load(file)



    train_dataset = Dataset.from_list(train[:1000])
    dev_dataset =  Dataset.from_list(train[1000:])
    test_dataset = Dataset.from_list(test)
    num_labels=2

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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


    padding = "max_length" if args.pad_to_max_length else False

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

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train dataset",
            num_proc=70,
            # batch_size=20000,
        )

    with accelerator.main_process_first():
        eval_dataset = dev_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dev_dataset.column_names,
            desc="Running tokenizer on test dataset",
        )


    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("aws_intern", experiment_config, init_kwargs={"wandb":{"name":args.output_dir.split("/")[1]}})

    # Get the metric function

    metric1 = evaluate.load("f1")
    metric2 = evaluate.load("precision")
    metric3 = evaluate.load("recall")
    metric4 = evaluate.load("accuracy")
    metric5 = evaluate.load("roc_auc")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    max_f1 =-1

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        samples_seen = 0
        preds = []
        refs = []
        txt = []
        txt2=[]
        all_scores=[]
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            if args.explainability:
                attentions = outputs.attentions
                predictions, references, attentions, localization_labels, scores = accelerator.gather((predictions, batch["labels"], attentions, batch["localization_label"], torch.softmax(outputs.logits)))
            else:
                predictions, references, logits, input_ids, response_input_ids = accelerator.gather((predictions, batch["labels"], outputs.logits, batch["input_ids"], batch["response_input_ids"]))
                scores = logits.softmax(dim=1)
                scores = [scores[ind][1] for ind in range(len(scores))]

                new_preds = []
                for each in scores:
                    if each > 0.2:
                        new_preds.append(1)
                    else:
                        new_preds.append(0)
                predictions = new_preds


            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                    scores = scores[: len(eval_dataloader.dataset) - samples_seen]
                    input_ids = input_ids[: len(eval_dataloader.dataset) - samples_seen]
                    response_input_ids = response_input_ids[: len(eval_dataloader.dataset) - samples_seen]
                    if args.explainability:
                        attentions = attentions[: len(eval_dataloader.dataset) - samples_seen]
                        localization_labels = localization_labels[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
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



            if args.explainability:
                attentions_new = []
                localization_labels_new = []
                for i in range(len(attentions)):

                    if not torch.equal(localization_labels[i].detach().cpu(), torch.Tensor([-1])[0]):
                        attentions_new.append(attentions[i].detach().cpu())
                        localization_labels_new.append(localization_labels[i].detach().cpu())

                metric4.add_batch(
                    predictions=attentions_new,
                    references=localization_labels_new,
                )




        eval_metric1 = metric1.compute()
        eval_metric2 = metric2.compute()
        eval_metric3 = metric3.compute()
        eval_metric5 = metric5.compute()
        logger.info(f"epoch {epoch}: {eval_metric1}")
        logger.info(f"epoch {epoch}: {eval_metric2}")
        logger.info(f"epoch {epoch}: {eval_metric3}")
        logger.info(f"epoch {epoch}: {eval_metric5}")
        if args.explainability:
            eval_metric4 = metric4.compute()
            logger.info(f"epoch {epoch}: {eval_metric4}")

        if args.with_tracking:
            logger.info(f"Train loss {total_loss.item() / len(train_dataloader)}")


        if args.with_tracking:
            accelerator.log(
                {
                    "f1": eval_metric1,
                    "precision": eval_metric2,
                    "recall": eval_metric3,
                    "ROC_AUC": eval_metric5,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
        if accelerator.is_main_process:
            if eval_metric5["roc_auc"]>max_f1:
                max_f1 = eval_metric5["roc_auc"]
                print("Writing!!!!!")
                with open('pred_ref_new.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["preds", "refs", "scores", "inputs", "inputs2"])
                    for ind in range(len(preds)):
                        to_write = [preds[ind], refs[ind], all_scores[ind], txt[ind], txt2[ind]]
                        writer.writerow(to_write)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.output_dir is not None:
            # all_results = {f"eval_{k}": v for k, v in eval_metric1.items()}
            with open(os.path.join(args.output_dir, "results_log.txt"), "a") as f:
                f.write(f"Epoch_{epoch}: Train Loss: {total_loss.item() / len(train_dataloader)} \n "
                        f"Eval F1: {eval_metric1} \n"
                        f"Eval Precision: {eval_metric2} \n"
                        f"Eval Recall: {eval_metric3} \n")

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )






if __name__ == "__main__":
    main()