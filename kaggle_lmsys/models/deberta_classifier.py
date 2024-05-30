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
    data: torch.Tensor,
    token_type_ids: torch.Tensor,
    model_a_resp_token_type_id: int,
    model_b_resp_token_tupe_id: int,
) -> torch.Tensor:
    """
    IN: [batch_size, text_length, embedding_size]
    OUT: [batch_size, embedding_size]

    :param x: [batch_size, text_length, embedding_size]
    :param token_type_ids: [batch_size, embedding_size]
    :return: [batch_size, embedding_size]
    """
    # [batch_size, text_length] -> [batch_size, text_length, embedding_size]
    token_type_ids = token_type_ids.unsqueeze(-1).expand(data.shape)
    mask_model_a_resp = torch.zeros(token_type_ids.shape, dtype=int).to(data.device)
    mask_model_a_resp[token_type_ids == model_a_resp_token_type_id] = 1
    mask_model_b_resp = torch.zeros(token_type_ids.shape, dtype=int).to(data.device)
    mask_model_b_resp[token_type_ids == model_b_resp_token_tupe_id] = 1

    model_a_rep = (data * mask_model_a_resp).mean(1)
    model_b_rep = (data * mask_model_b_resp).mean(1)
    resp = model_a_rep - model_b_rep

    return resp


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
    def __init__(self, config: CustomizedDetertaConfig):
        super().__init__()
        self.config = config
        # self.model_a_resp_token_type_id = self.config.model_a_resp_token_type_id
        # self.model_b_resp_token_type_id = self.config.model_b_resp_token_type_id
        self.model_a_resp_token_type_id = 1
        self.model_b_resp_token_type_id = 2
        self.dense = nn.Linear(self.config.pooler_hidden_size, self.config.pooler_hidden_size)
        self.dropout = nn.Dropout(self.config.pooler_dropout)

    def forward(self, data: torch.Tensor, token_type_ids: torch.Tensor):
        resp = get_diff_of_two_response(
            data, token_type_ids, self.model_a_resp_token_type_id, self.model_b_resp_token_type_id,
        )
        resp = self.dropout(resp)
        resp = self.dense(resp)
        resp = ACT2FN[self.config.pooler_hidden_act](resp)
        return resp

    @property
    def output_dim(self):
        return self.config.hidden_size


class CustomizedDetertaClassifier(DebertaForSequenceClassification):
    config_class = CustomizedDetertaConfig

    def __init__(
        self,
        config: CustomizedDetertaConfig,
    ):
        super().__init__(config)
        self.pooler = ModelResDiffPooler(config)

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
        pooled_output = self.pooler(encoder_layer, token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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


AutoConfig.register("CustomizedDetertaClassifier", CustomizedDetertaConfig)
AutoModelForSequenceClassification.register(CustomizedDetertaConfig, CustomizedDetertaClassifier)
