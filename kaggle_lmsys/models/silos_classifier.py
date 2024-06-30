from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.activations import ACT2FN
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import MPNetConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mpnet.modeling_mpnet import MPNetModel


def filter_embeddings_based_on_token_type_ids(
    embeddings: torch.Tensor,
    token_type_ids: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    outputs = torch.vstack(
        [
            torch.masked_select(_embeddings, _token_type_ids == token_id)
            for _embeddings, _token_type_ids in zip(embeddings, token_type_ids)
        ]
    )
    return outputs


class SiloClassifierConfig(MPNetConfig):
    model_type = "LMSYSSilosClassifier"

    def __init__(
        self,
        *args,
        pretrained_mpnet_name: str = "sentence-transformers/all-mpnet-base-v2",
        prompt_token_type_id: int = 0,
        model_one_resp_token_type_id: int = 1,
        model_two_resp_token_type_id: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pretrained_mpnet_name = pretrained_mpnet_name
        self.prompt_token_type_id = prompt_token_type_id
        self.model_one_resp_token_type_id = model_one_resp_token_type_id
        self.model_two_resp_token_type_id = model_two_resp_token_type_id


class MPNETSilos(nn.Module):
    def __init__(self, config: SiloClassifierConfig):
        super().__init__()
        self.config = config
        self.mpnet: MPNetModel = MPNetModel(self.config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """

        :param input_ids: [batch_size, num_words]
        :param attention_mask: [batch_size, num_words]
        :param token_type_ids: [batch_size, num_words]
        :return: [batch, 4 * num_words, size_embeddings]
        """
        input_ids_one = filter_embeddings_based_on_token_type_ids(
            input_ids,
            token_type_ids,
            self.config.model_one_resp_token_type_id,
        )
        attention_mask_one = filter_embeddings_based_on_token_type_ids(
            attention_mask,
            token_type_ids,
            self.config.model_one_resp_token_type_id,
        )
        input_ids_two = filter_embeddings_based_on_token_type_ids(
            input_ids,
            token_type_ids,
            self.config.model_two_resp_token_type_id,
        )
        attention_mask_two = filter_embeddings_based_on_token_type_ids(
            attention_mask,
            token_type_ids,
            self.config.model_two_resp_token_type_id,
        )
        silo_one_outputs = self.mpnet(
            input_ids_one,
            attention_mask=attention_mask_one,
        )
        silo_two_outputs = self.mpnet(
            input_ids_two,
            attention_mask=attention_mask_two,
        )
        silo_one_last_hidden_state = silo_one_outputs.last_hidden_state.mean(1)
        silo_two_last_hidden_state = silo_two_outputs.last_hidden_state.mean(1)

        hidden_states = torch.concatenate(
            [
                silo_one_last_hidden_state,
                silo_two_last_hidden_state,
                silo_one_last_hidden_state - silo_two_last_hidden_state,
            ],
            axis=1,
        )
        silo_outputs = silo_one_outputs  # FIXME
        silo_outputs.last_hidden_state = hidden_states
        return silo_outputs


class ModelResponsePooler(nn.Module):
    def __init__(
        self, config: SiloClassifierConfig,
    ):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(self.config.hidden_size * 3, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(0.2)

    @property
    def output_dim(self) -> int:
        return self.config.hidden_size

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.dense(data)
        data = self.dropout(data)
        data = self.norm(data)
        data = ACT2FN[self.config.hidden_act](data)
        return data


class LMSYSSilosClassifier(PreTrainedModel):
    config_class = SiloClassifierConfig

    def __init__(self, config: SiloClassifierConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = self.config.num_labels
        self.silos: MPNETSilos = MPNETSilos(self.config)
        self.model_resp_pool = ModelResponsePooler(self.config)
        self.classifier = nn.Linear(self.model_resp_pool.output_dim, self.config.num_labels)

    def update_silos_mpnet_with_pretrained_model(self, pretrained_model_name: str) -> None:
        self.silos.mpnet = MPNetModel.from_pretrained(pretrained_model_name)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        outputs = self.silos(
            input_ids,
            attention_mask,
            token_type_ids,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = self.model_resp_pool(hidden_states)
        logits = self.classifier(hidden_states)

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

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


AutoConfig.register("LMSYSSilosClassifier", SiloClassifierConfig)
AutoModelForSequenceClassification.register(SiloClassifierConfig, LMSYSSilosClassifier)
