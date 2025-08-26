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
Text Chunking Utilities for Long Context Processing

This module provides utilities for splitting long text inputs into smaller, manageable chunks
that can be processed by transformer models with limited context windows. It supports various
chunking strategies including overlapping chunks, sentence-based splitting, and paired 
context-response chunking.

Key Features:
- Overlapping chunk generation with configurable stride
- Sentence-based chunking for response text
- Context-response pairing for NLI-style processing
- Proper handling of special tokens (CLS, SEP, PAD)
- Flexible padding strategies

Author: Siyi Liu et al.
"""
  
from typing import Optional, List, Tuple, Union
import logging

import torch
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerBase


def transform_list_of_text_pairs(
    texts1: list[str],
    texts2 : list[str],
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    num_chunks1: int,
    num_chunks2: int,
    pad_last: bool,
    pad_original: bool,
    maximal_text_length: int,
    split_sent: bool,
    sent_length: int,
    pair_chunks: bool
) -> BatchEncoding:
    model_inputs1 = []
    model_inputs2 = []
    for i in range(len(texts1)):
        id, mask = transform_single_text(texts1[i], tokenizer, chunk_size, stride, minimal_chunk_length, num_chunks1,
                              maximal_text_length, pad_last, pad_original)
        model_inputs1.append((id, mask))
        id2, mask2 = transform_single_text(texts2[i], tokenizer, chunk_size, stride, minimal_chunk_length, num_chunks2, maximal_text_length, pad_last, pad_original, split_sent, sent_length)
        if split_sent:
            if len(id2)>=num_chunks2:
                id2 = id2[:num_chunks2]
                mask2 = mask2[:num_chunks2]
            else:
                chunks_to_pad = num_chunks2-len(id2)
                ids_pad = torch.full((1, len(id2[0])), 1)
                att_pad = torch.full((1, len(id2[0])), 0)
                for _ in range(chunks_to_pad):
                    id2 = torch.cat((id2,ids_pad))
                    mask2 = torch.cat((mask2, att_pad))

        model_inputs2.append((id2, mask2))




    input_ids = [model_input[0][:num_chunks1] for model_input in model_inputs1]
    attention_mask = [model_input[1][:num_chunks1] for model_input in model_inputs1]

    print(len(input_ids))

    input_ids_2 = [model_input[0][:num_chunks2] for model_input in model_inputs2]
    attention_mask_2 = [model_input[1][:num_chunks2] for model_input in model_inputs2]

    if split_sent:

        if pair_chunks:
            all_paired_input_ids = []
            all_paired_attention_mask=[]
            for i in range(len(input_ids)):
                paired_input_ids = []
                paired_attention_mask = []
                for j in range(len(input_ids[i])):
                    for k in range(len(input_ids_2[i])):
                        paired_ids, paired_mask = pair_context_response_chunks(input_ids[i][j], attention_mask[i][j], input_ids_2[i][k], attention_mask_2[i][k])
                        paired_input_ids.append(paired_ids)
                        paired_attention_mask.append(paired_mask)
                all_paired_input_ids.append(paired_input_ids)
                all_paired_attention_mask.append(paired_attention_mask)



            tokens = {"input_ids": all_paired_input_ids, "attention_mask": all_paired_attention_mask}
            return BatchEncoding(tokens)

        else:
            tokens = {"input_ids": input_ids, "attention_mask": attention_mask, "response_input_ids": input_ids_2, "response_attention_mask": attention_mask_2}
            return BatchEncoding(tokens)

    for i in range(len(input_ids_2)):
        input_ids[i] = torch.cat((input_ids[i], input_ids_2[i]))
        attention_mask[i] = torch.cat((attention_mask[i], attention_mask_2[i]))

    tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
    return BatchEncoding(tokens)

def transform_list_of_text(
    texts1: list[str],
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    num_chunks1: int,
    pad_last: bool,
    pad_original:bool,
    maximal_text_length: int,

) -> BatchEncoding:
    model_inputs1 = []
    for i in range(len(texts1)):
        id, mask = transform_single_text(texts1[i], tokenizer, chunk_size, stride, minimal_chunk_length, num_chunks1,
                              maximal_text_length, pad_last, pad_original)
        model_inputs1.append((id, mask))

    input_ids = [model_input[0] for model_input in model_inputs1]
    attention_mask = [model_input[1] for model_input in model_inputs1]

    tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
    return BatchEncoding(tokens)

def transform_single_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    num_chunks: int,
    maximal_text_length: Optional[int],
    pad_last: bool,
    pad_original: bool,
    split_sent= False,
    sent_length=0
) -> tuple[Tensor, Tensor]:
    """Transforms (the entire) text to model input of BERT model."""
    if split_sent:
        sentences = text.split(". ")
        tokens = tokenizer(
            sentences, padding="max_length", max_length=sent_length, truncation=True, return_tensors="pt")
        return tokens["input_ids"], tokens["attention_mask"]

    elif pad_last:
        tokens = tokenize_text_with_truncation(text, tokenizer, maximal_text_length, chunk_size)
    else:
        tokens = tokenize_whole_text(text, tokenizer)



    input_id_chunks = split_overlapping(tokens["input_ids"][0], chunk_size, stride, minimal_chunk_length, num_chunks,
                                        maximal_text_length, pad_last=pad_last, pad_original=pad_original)
    mask_chunks = split_overlapping(tokens["attention_mask"][0], chunk_size, stride, minimal_chunk_length, num_chunks,
                                    maximal_text_length, pad_last=pad_last, pad_original=pad_original)

    if pad_last:
        add_special_tokens_at_beginning_and_end_pad_last(input_id_chunks, mask_chunks)
    else:
        add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
        add_padding_tokens(input_id_chunks, mask_chunks, chunk_size)


    input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
    # print(input_ids.shape)
    return input_ids, attention_mask


def tokenize_whole_text(text: str, tokenizer: PreTrainedTokenizerBase) -> BatchEncoding:
    """Tokenizes the entire text without truncation and without special tokens."""
    tokens = tokenizer(text, add_special_tokens=False, truncation=False, return_tensors="pt")
    return tokens


def tokenize_text_with_truncation(
    text: str, tokenizer: PreTrainedTokenizerBase, maximal_text_length: int, chunk_size:int,
) -> BatchEncoding:
    """Tokenizes the text with truncation to maximal_text_length and without special tokens."""

    max_length = int(maximal_text_length - (maximal_text_length//chunk_size)*2)

    tokens = tokenizer(
        text, add_special_tokens=False, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt"
    )
    return tokens



def split_overlapping(tensor: Tensor, chunk_size: int, stride: int, minimal_chunk_length: int, num_chunks: int, maximal_text_length: int, pad_last: bool, pad_original: bool) -> list[Tensor]:
    """Helper function for dividing 1-dimensional tensors into overlapping chunks."""
    if pad_original:
        chunk_size_without_pad = len(tensor) // num_chunks + 1
        stride_without_pad = chunk_size_without_pad
        result = [tensor[i: i + chunk_size_without_pad] for i in range(0, len(tensor), stride_without_pad)]
    elif pad_last:
        chunk_size = chunk_size-2
        result = [tensor[i: i + chunk_size] for i in range(0, len(tensor), chunk_size)]


    else:

        chunk_size_without_pad = len(tensor) // num_chunks


        result = [tensor[i* chunk_size_without_pad : (i+1) * chunk_size_without_pad] for i in range(0, num_chunks-1)]
        result.append(tensor[chunk_size_without_pad*(num_chunks-1):])

    return result

def add_special_tokens_at_beginning_and_end(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
    """
    Adds special CLS token (token id = 101) at the beginning.
    Adds SEP token (token id = 102) at the end of each chunk.
    Adds corresponding attention masks equal to 1 (attention mask is boolean).
    """
    if len(input_id_chunks) == 0:
        input_id_chunks.append(torch.Tensor([0, 2]))
        mask_chunks.append(torch.Tensor([1, 1]))
        return
    for i in range(len(input_id_chunks)):
        # adding CLS (token id 101) and SEP (token id 102) tokens
        input_id_chunks[i] = torch.cat([Tensor([0]), input_id_chunks[i], Tensor([2])])
        # adding attention masks  corresponding to special tokens
        mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i], Tensor([1])])

def add_special_tokens_at_beginning_and_end_pad_last(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
    """
    Adds special CLS token (token id = 101) at the beginning.
    Adds SEP token (token id = 102) at the end of each chunk.
    Adds corresponding attention masks equal to 1 (attention mask is boolean).
    """
    if len(input_id_chunks) == 0:
        input_id_chunks.append(torch.Tensor([0, 2]))
        mask_chunks.append(torch.Tensor([1, 1]))
        return

    for i in range(len(input_id_chunks)):
        # adding CLS (token id 101) and SEP (token id 102) tokens

        if sum(input_id_chunks[i]) == len(input_id_chunks[i]):
            input_id_chunks[i] = torch.cat([Tensor([1]), input_id_chunks[i], Tensor([1])])
            mask_chunks[i] = torch.cat([Tensor([0]), mask_chunks[i], Tensor([0])])

        else:
            if mask_chunks[i][-1]==1:

                input_id_chunks[i] = torch.cat([Tensor([0]), input_id_chunks[i], Tensor([2])])
                # adding attention masks  corresponding to special tokens
                mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i], Tensor([1])])
            else:
                for j in range(len(input_id_chunks[i])):
                    if torch.equal(mask_chunks[i][j], torch.Tensor([0])[0]):
                        input_id_chunks[i] = torch.cat([Tensor([0]), input_id_chunks[i][:j], Tensor([2]), input_id_chunks[i][j:]])
                        mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i][:j], Tensor([1]), mask_chunks[i][j:]])
                        break


def add_padding_tokens(input_id_chunks: list[Tensor], mask_chunks: list[Tensor], chunk_size: int) -> None:
    """Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens."""
    for i in range(len(input_id_chunks)):
        # get required padding length
        pad_len = chunk_size - input_id_chunks[i].shape[0]
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([input_id_chunks[i], Tensor([1] * pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], Tensor([0] * pad_len)])


def stack_tokens_from_all_chunks(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Reshapes data to a form compatible with BERT model input."""
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    return input_ids.long(), attention_mask.int()

def check_split_parameters_consistency(chunk_size: int, stride: int, minimal_chunk_length: int, num_chunks, token_length, pad_last) -> None:
    if pad_last:
        pass
    else:
        chunk_size_without_pad = token_length//num_chunks + 1
        if chunk_size_without_pad > chunk_size:
            raise Exception("Size chunk without pad is longer than chunk size!")


def pair_context_response_chunks(input_ids, attention_mask, input_ids2, attention_mask2):
    if torch.equal(torch.sum(attention_mask), torch.Tensor([0])[0]) or torch.equal(torch.sum(attention_mask2), torch.Tensor([0])[0]):
        shape = input_ids.shape[0] + input_ids2.shape[0]
        paired_input_ids = torch.ones([shape])
        paired_attention_mask = torch.zeros([shape])
        return paired_input_ids.int(), paired_attention_mask.int()


    if not torch.equal(input_ids[-1], torch.Tensor([2])[0]):
        ones = (input_ids == 1.).sum(dim=0)
        for i in range(len(input_ids)):
            if torch.equal(input_ids[i], torch.Tensor([1])[0]):
                paired_input_ids = torch.cat([input_ids[:i], input_ids2, torch.ones((ones))], dim=0)
                paired_attention_mask = torch.cat([attention_mask[:i], attention_mask2, torch.zeros((ones))], dim=0)
                return paired_input_ids.int(), paired_attention_mask.int()



    else:
        input_ids2[0] = torch.Tensor([2])
        paired_input_ids = torch.cat([input_ids, input_ids2], dim=0)
        paired_attention_mask = torch.cat([attention_mask, attention_mask2], dim=0)

        return paired_input_ids.int(), paired_attention_mask.int()


