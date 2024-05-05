from dataclasses import dataclass
from typing import Dict


@dataclass
class DataConfig:
    train_data_path: str
    test_data_path: str

    @staticmethod
    def from_config(data_config: Dict):
        return DataConfig(
            data_config["train"]["path"],
            data_config["test"]["path"],
        )


@dataclass
class PartitionSchemaCV:
    folds: int

    @staticmethod
    def from_config(data_config: Dict):
        partition = data_config["partition"]
        return PartitionSchemaCV(int(partition["cv"]))


@dataclass
class PartitionSchemaTVH:
    train_pct: float
    train_label: int
    validation_pct: float
    validation_label: int
    holdout_pct: float
    holdout_label: int

    @staticmethod
    def from_config(data_config: Dict):
        partition = data_config["partition"]["tvh"]
        return PartitionSchemaTVH(
            float(partition["train"]["pct"]),
            int(partition["train"]["label"]),
            float(partition["validation"]["pct"]),
            int(partition["validation"]["label"]),
            float(partition.get("holdout", {}).get("pct", 0.0)),
            int(partition.get("holdout", {}).get("label", -1)),
        )


@dataclass
class TrainConfig:
    random_seed: int
    model_output_path: str
    enable_grad_scale: bool
    partition_cv: PartitionSchemaCV
    partition_tvh: PartitionSchemaTVH
    num_epoch: int
    num_gradient_update_batch: int
    train_batch_size: int
    eval_batch_size: int
    holdout_batch_size: int
    learning_rate: float
    min_learning_rate: float

    @staticmethod
    def from_config(data_config: Dict):
        return TrainConfig(
            random_seed=int(data_config["random_seed"]),
            model_output_path=data_config["model_output_path"],
            enable_grad_scale=bool(data_config["enable_grad_scale"]),
            partition_cv=PartitionSchemaCV.from_config(data_config),
            partition_tvh=PartitionSchemaTVH.from_config(data_config),
            num_epoch=int(data_config["num_epoch"]),
            num_gradient_update_batch=int(data_config["num_gradient_update_batch"]),
            train_batch_size=int(data_config["train_batch_size"]),
            eval_batch_size=int(data_config["eval_batch_size"]),
            holdout_batch_size=int(data_config["holdout_batch_size"]),
            learning_rate=float(data_config["learning_rate"]),
            min_learning_rate=float(data_config["min_learning_rate"]),
        )


@dataclass
class ModelConfig:
    model_name: str
    num_classes: int

    @staticmethod
    def from_config(data_config: Dict):
        return ModelConfig(
            model_name=data_config["model_name"],
            num_classes=data_config.get("num_classes", 0),
        )


@dataclass
class TokenizationConfig:
    model_name: str
    max_length: int
    label_name: str
    context_mask_field: str
    answer_mask_field: str
    tokenization_output_fields: set[str]
    special_token_answer_start: str
    special_token_answer_end: str
    special_token_context_start: str
    special_token_context_end: str
    special_token_context_sep: str

    @staticmethod
    def from_config(data_config: Dict):
        return TokenizationConfig(
            model_name=data_config["model_name"],
            max_length=int(data_config["max_length"]),
            label_name=data_config["label_name"],
            tokenization_output_fields=set(
                data_config["tokenization_output_fields"].split(",")
            ),
            context_mask_field=data_config["context_mask_field"],
            answer_mask_field=data_config["answer_mask_field"],
            special_token_answer_start=data_config["special_token_answer_start"],
            special_token_answer_end=data_config["special_token_answer_end"],
            special_token_context_start=data_config["special_token_context_start"],
            special_token_context_end=data_config["special_token_context_end"],
            special_token_context_sep=data_config["special_token_context_sep"],
        )