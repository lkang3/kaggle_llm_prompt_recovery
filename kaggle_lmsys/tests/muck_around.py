import pytest
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.classifier_lgbm_pipeline import LGBMClassifierPipeline
from kaggle_lmsys.models.embedding_flow_deterta import DetertaEmbeddingLMSYSFlow
from kaggle_lmsys.models.embedding_flow_word2vec import W2VEmbeddingLMSYSFlow


SEED = 123


@pytest.fixture
def random_state() -> np.random.RandomState:
    return np.random.RandomState(SEED)


@pytest.fixture
def data() -> pd.DataFrame:
    data = pd.read_csv("/home/lkang/Downloads/lmsys-chatbot-arena/train.csv")
    return data.loc[: 10, :]


class TestW2VEmbeddingFlow:
    def test(self, data: pd.DataFrame) -> None:
        config = {
            "model_name": "glove-twitter-200",
            "model_path": "/home/lkang/Downloads/w2v_glove-twitter-200",
            "embedding_aggregator": "mean",
        }
        flow = W2VEmbeddingLMSYSFlow(config)
        flow.fit_and_inference(data)


class TestDetertaEmbeddingFlow:
    def test(self, data: pd.DataFrame) -> None:
        config = {
            "tokenizer_name": "microsoft/deberta-base",
            "model_name": "microsoft/deberta-base",
            "tokenizer_output_path": "/home/lkang/Downloads/token_llm_outputs",
            "model_output_path": "/home/lkang/Downloads/token_llm_outputs",
            "data": {
                "prompt": "prompt",
                "resp_a": "response_a",
                "resp_b": "response_b",
            },
            "max_prompt_token_length": 64,
            "max_resp_a_token_length": 255,
            "max_resp_b_token_length": 255,
        }
        flow = DetertaEmbeddingLMSYSFlow(config)
        flow.inference(data)


class TestLGBMClassifierPipeline:
    def test(self, random_state: np.random.RandomState) -> None:
        num_classes = 3
        config = {
            "lgbm": {
                "params":
                    {
                        'n_estimators': 1000,
                        'max_depth': 4,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'objective': 'multiclass',
                        'num_class': random_state,
                        'metric': 'multi_logloss',
                        'random_state': SEED,
                        'learning_rate': 0.003,
                    },
                "eval_pct": 0.2,
                "early_stopping": 100,
            },
            "seed": SEED,
            "output_path": "/home/lkang/Downloads/lgbm_model.pkl",
        }

        model_data = ModelData(
            data_types=np.array([DataType.NUM, DataType.NUM]),
            x=random_state.randn(100, 2),
            y=random_state.randint(0, num_classes, 100),
        )
        model = LGBMClassifierPipeline(config)
        model.fit(model_data)
        model.predict_proba(model_data)
