from pathlib import Path

import numpy as np

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import GemmaTokenizerFast
from transformers import TrainingArguments
from transformers import Trainer

from kaggle_lmsys.models.entities import ModelData


class ModelTypePredAIFlow:
    def __init__(self) -> None:
        pass

    def fit(self, data: ModelData) -> "ModelTypePredAIFlow":
        return self

    def inference(self, data: ModelData) -> np.ndarray:
        pass


class ModelTypeGenAIFlow:
    def __init__(
        self,
        tokenizer_config_path: Path,
        llm_model_config_path: Path,
    ) -> None:
        self._tokenizer_config_path = tokenizer_config_path
        self._llm_model_config_path = llm_model_config_path

    def _load_tokenizer(self) -> object:
        tokenize = AutoTokenizer.from_pretrained()

    def _load_model(self) -> None:
        pass

    def _save_tokenizer(self) -> None:
        pass

    def _save_model(self) -> None:
        pass

    def fit(self, data: ModelData) -> "ModelTypeGenAIFlow":
        pass

    def inference(self, data: ModelData) -> np.ndarray:
        pass
