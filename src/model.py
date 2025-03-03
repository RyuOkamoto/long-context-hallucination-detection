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
  
import torch
import torch.utils.checkpoint
from packaging import version
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
from torch.nn.functional import normalize
import numpy as np

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import (
    ModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions
)
from torch import Tensor
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaIntermediate, RobertaOutput, RobertaPreTrainedModel, RobertaClassificationHead, RobertaEncoder, RobertaPooler, RobertaEmbeddings
from transformers.activations import gelu
from transformers import PretrainedConfig
import copy

class RobertaModelOurs(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    # Copied from transformers.models.clap.modeling_clap.ClapTextModel.__init__ with ClapText->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    # Copied from transformers.models.clap.modeling_clap.ClapTextModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class RobertaClassificationHeadOurs(nn.Module):
    """Head for sentence-level classification tasks."""
    # The only difference of this to source code is that we comment out the first line in forward
    # because in pad_last the number of chunks can be different so we take the CLS first outside
    # of this forward function.

    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassificationOurs(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModelOurs(config, add_pooling_layer=False)
        

        if self.config.attention_encoder:
            config_oneLayer = copy.deepcopy(config)
            config_oneLayer.num_hidden_layers = 1
            self.roberta_oneLayer = RobertaModelOurs(config, add_pooling_layer=False)

        # self.roberta_oneLayer.embeddings.word_embeddings.weight.requires_grad = False
        if config.pad_last:
            self.classifier = RobertaClassificationHeadOurs(config)
        else:
            self.classifier = RobertaClassificationHead(config)
        self.split = config.split


        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        localization_label = None,
        response_input_ids = None,
        response_attention_mask = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        input_ids: a list of int with size batch_size * num_chunks * chunk_length
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        batch_size = len(input_ids)





        if self.split:

            all_sequence_output = []

            all_self_attentions = []

            for i in range(len(input_ids)):

                if self.config.pad_last:

                    chunks_input_ids, chunks_attention_mask, num_contexts_chunks =self.remove_chunks_with_only_pads(input_ids[i], attention_mask[i])

                    outputs = self.roberta(
                        chunks_input_ids,
                        attention_mask=chunks_attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )

                    sequence_output = outputs[0]


                    if self.config.split_sent:
                        if not self.config.pair_chunks:
                            response_chunks_input_ids, response_chunks_attention_mask, num_response_chunks = self.remove_chunks_with_only_pads(response_input_ids[i],
                                                                                                        response_attention_mask[i])
                            outputs_response = self.roberta(
                                response_chunks_input_ids,
                                attention_mask=response_chunks_attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                            )
                            sequence_output2 = outputs_response[0]

                else:
                    outputs = self.roberta(
                        input_ids[i],
                        attention_mask=attention_mask[i],
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                    sequence_output = outputs[0]
                if self.config.attention_encoder:
                    # num_chunks * chunk_size * hidden_size ->  1 * num_chunks * hidden_size

                    chunk_cls_embeds = sequence_output[:, 0, :].unsqueeze(0) #.reshape(1, -1, self.config.hidden_size)
                    if self.config.split_sent:
                        if not self.config.pair_chunks:
                            chunk_cls_embeds2 = sequence_output2[:, 0, :].unsqueeze(0)
                            chunk_cls_embeds = torch.cat([chunk_cls_embeds, chunk_cls_embeds2], dim=1)

                    CLS_embed = self.roberta_oneLayer.embeddings.word_embeddings(Tensor([0]).long().to(self.device))

                    EOS_embed = self.roberta_oneLayer.embeddings.word_embeddings(Tensor([2]).long().to(self.device))

                    SEP_embed = self.roberta_oneLayer.embeddings.word_embeddings(Tensor([2]).long().to(self.device))


                    if self.config.split_inputs:
                        if self.config.add_sep:
                            chunk_embeds_with_new_tokens = torch.cat([CLS_embed, chunk_cls_embeds[0][:num_contexts_chunks],SEP_embed, chunk_cls_embeds[0][num_contexts_chunks:], EOS_embed]).unsqueeze(
                                0)
                        else:
                            chunk_embeds_with_new_tokens = torch.cat([CLS_embed, chunk_cls_embeds[0], EOS_embed]).unsqueeze(0)
                    else:
                        chunk_embeds_with_new_tokens = torch.cat([CLS_embed, chunk_cls_embeds[0], EOS_embed]).unsqueeze(0)

                    outputs2 = self.roberta_oneLayer(
                        None,
                        attention_mask=None,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=chunk_embeds_with_new_tokens,
                        output_attentions=self.config.explainability,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )

                    if self.config.explainability:
                        all_self_attentions += [outputs2.attentions[0].mean(-1).argmax(-1)[-1][-1]]

                    average_chunk_outputs = outputs2[0]

                else:
                    average_chunk_outputs = torch.mean(sequence_output, dim=0).reshape((1, -1, self.config.hidden_size))



                all_sequence_output.append(average_chunk_outputs[:,0,:])




            all_sequence_output = torch.cat(all_sequence_output, dim=0)

            logits = self.classifier(all_sequence_output)
        else:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)


        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(self.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # print(logits.view(-1, self.num_labels))
                # print(labels.view(-1))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if self.config.explainability:

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=torch.stack(all_self_attentions),
            )

        else:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                # attentions=outputs2.attentions,
            )
    def pool_chunks(self, input_ids, attention_mask) -> Tensor:

        number_of_chunks = [len(x) for x in input_ids]

        # concatenate all input_ids into one batch

        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x)

        input_ids_combined_tensors = torch.stack([torch.tensor(x).to(self.device) for x in input_ids_combined])

        # concatenate all attention masks into one batch

        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x)

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(self.device) for x in attention_mask_combined]
        )

        # return pooled_logits

        return input_ids_combined_tensors, attention_mask_combined_tensors

    def remove_chunks_with_only_pads(self, input_ids, attention_mask):
        all_pads_chunks_ind = []
        for j in range(len(attention_mask)):
            if torch.equal(torch.sum(attention_mask[j]), torch.Tensor([0])[0].to(self.device)):
                all_pads_chunks_ind.append(j)

        chunks_input_ids = input_ids
        chunks_attention_mask = attention_mask

        if len(all_pads_chunks_ind) != 0 and all_pads_chunks_ind[0] < self.config.num_chunks_context:
            num_contexts_chunks = all_pads_chunks_ind[0]
        else:
            num_contexts_chunks = self.config.num_chunks_context


        for index in sorted(all_pads_chunks_ind, reverse=True):
            if index == len(chunks_input_ids):
                chunks_input_ids = chunks_input_ids[:index]
                chunks_attention_mask = chunks_attention_mask[:index]
            else:
                chunks_input_ids = torch.cat([chunks_input_ids[:index], chunks_input_ids[index + 1:]], dim=0)
                chunks_attention_mask = torch.cat([chunks_attention_mask[:index], chunks_attention_mask[index + 1:]], dim=0)

        return chunks_input_ids, chunks_attention_mask, num_contexts_chunks