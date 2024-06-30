from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import DebertaConfig, DebertaForSequenceClassification
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification


def load_class(module_name: str, class_name: str) -> None:
    module = __import__(module_name)
    return getattr(module, class_name)


def get_diff_of_two_response(
    hidden_states: torch.Tensor,
    token_type_ids: torch.Tensor,
    model_resp_len_diff_type_id: int,
    model_a_resp_token_type_id: int,
    model_b_resp_token_tupe_id: int,
) -> torch.Tensor:
    """
    IN: [batch_size, text_length, embedding_size]
    OUT: [batch_size, embedding_size]

    :param hidden_states: [batch_size, text_length, embedding_size]
    :param token_type_ids: [batch_size, text_length]
    :return: [batch_size, embedding_size]
    """
    # [batch_size, text_length] -> [batch_size, text_length, embedding_size]
    token_type_ids = token_type_ids.unsqueeze(-1).expand(hidden_states.shape)
    mask_model_a_resp = torch.zeros(token_type_ids.shape, dtype=int).to(hidden_states.device)
    mask_model_a_resp[token_type_ids == model_a_resp_token_type_id] = 1
    mask_model_b_resp = torch.zeros(token_type_ids.shape, dtype=int).to(hidden_states.device)
    mask_model_b_resp[token_type_ids == model_b_resp_token_tupe_id] = 1
    mask_model_resp_len_diff = torch.zeros(token_type_ids.shape, dtype=int).to(hidden_states.device)
    mask_model_resp_len_diff[token_type_ids == model_resp_len_diff_type_id] = 1


    model_a_resp = (hidden_states * mask_model_a_resp).mean(1)
    model_b_resp = (hidden_states * mask_model_b_resp).mean(1)
    model_resp_len_diff = (hidden_states * mask_model_resp_len_diff).mean(1)
    model_a_b_resp_diff = model_a_resp - model_b_resp

    return model_a_resp, model_b_resp, model_a_b_resp_diff, model_resp_len_diff


class CustomizedDetertaConfig(DebertaConfig):
    model_type = "CustomizedDetertaClassifier"

    def __init__(
        self,
        *args,
        custom_pooler: Optional[nn.Module] = None,
        prompt_token_type_id: int = 0,
        model_a_resp_token_type_id: int = 1,
        model_b_resp_token_type_id: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.custom_pooler = custom_pooler
        self.prompt_token_type_id = prompt_token_type_id
        self.model_a_resp_token_type_id = model_a_resp_token_type_id
        self.model_b_resp_token_type_id = model_b_resp_token_type_id


class ModelResDiffPooler(nn.Module):
    def __init__(
        self,
        config: CustomizedDetertaConfig,
        model_resp_len_diff_type_id: int,
        model_a_resp_token_type_id: int,
        model_b_resp_token_type_id: int,
    ):
        super().__init__()
        self.config = config
        self.model_resp_len_diff_type_id = model_resp_len_diff_type_id
        self.model_a_resp_token_type_id = model_a_resp_token_type_id
        self.model_b_resp_token_type_id = model_b_resp_token_type_id
        self.dense_one = nn.Linear(
            self.config.pooler_hidden_size * 4, self.config.pooler_hidden_size * 2,
        )
        self.dense_two = nn.Linear(
            self.config.pooler_hidden_size * 2, self.config.pooler_hidden_size,
        )
        self.dropout = nn.Dropout(self.config.pooler_dropout)
        self.norm = nn.LayerNorm(self.config.hidden_size)

    def forward(self, data: torch.Tensor, token_type_ids: torch.Tensor):
        model_a_resp, model_b_resp, model_resp_diff, model_resp_len_diff = get_diff_of_two_response(
            data,
            token_type_ids,
            self.model_resp_len_diff_type_id,
            self.model_a_resp_token_type_id,
            self.model_b_resp_token_type_id,
        )
        hidden_states = torch.concatenate(
            [model_a_resp, model_b_resp, model_resp_diff, model_resp_len_diff], axis=-1
        )
        resp = self.dense_one(hidden_states)
        resp = self.dense_two(resp)
        resp = self.dropout(resp)
        resp = self.norm(resp)
        resp = ACT2FN[self.config.pooler_hidden_act](resp)
        return resp

    @property
    def output_dim(self):
        return self.config.hidden_size


class EasyCustomizableDebertaForSequenceClassification(DebertaForSequenceClassification):
    def _get_logits(self, encoder_layer, token_type_ids, attention_mask):
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        logits = self._get_logits(encoder_layer, token_type_ids, attention_mask)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class CustomizedDetertaClassifier(EasyCustomizableDebertaForSequenceClassification):
    config_class = CustomizedDetertaConfig

    def __init__(
        self,
        config: CustomizedDetertaConfig,
    ):
        super().__init__(config)
        model_resp_len_diff_type_id = config.prompt_token_type_id
        model_a_resp_token_type_id = config.model_a_resp_token_type_id
        model_b_resp_token_type_id = config.model_b_resp_token_type_id

        self.pooler = ModelResDiffPooler(
            config,
            model_resp_len_diff_type_id,
            model_a_resp_token_type_id,
            model_b_resp_token_type_id,
        )

    def _get_logits(self, encoder_layer, token_type_ids, attention_mask):
        pooled_output = self.pooler(encoder_layer, token_type_ids)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


AutoConfig.register("CustomizedDetertaClassifier", CustomizedDetertaConfig)
AutoModelForSequenceClassification.register(CustomizedDetertaConfig, CustomizedDetertaClassifier)
