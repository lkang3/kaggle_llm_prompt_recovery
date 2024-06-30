import pytest

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import MPNetTokenizerFast

from kaggle_lmsys.models.silos_classifier import LMSYSSilosClassifier
from kaggle_lmsys.models.silos_classifier import SiloClassifierConfig
from kaggle_lmsys.utils import tokenization_separate
from kaggle_lmsys.utils import Collator


class TestLMSYSSilosClassifier:
    @pytest.fixture
    def model_name(self) -> str:
        return "sentence-transformers/all-mpnet-base-v2"

    @pytest.fixture
    def datasets(
        self,
        input_dataframe: pd.DataFrame,
        model_name: str,
    ) -> Dataset:
        dataset = Dataset.from_pandas(input_dataframe)
        dataset.cleanup_cache_files()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenization_args = {
            "tokenizer": tokenizer,
            "prompt_field": "prompt",
            "resp_a_field": "response_a",
            "resp_b_field": "response_b",
            "max_token_length": 574,
            "max_prompt_token_length": 64,
            "max_resp_a_token_length": 255,
            "max_resp_b_token_length": 255,
            "target_field": "labels",
        }
        return dataset.map(
            function=tokenization_separate,
            batched=False,
            fn_kwargs=tokenization_args,
            remove_columns=dataset.column_names,
        )

    @pytest.fixture
    def model_name(self) -> str:
        return "sentence-transformers/all-mpnet-base-v2"

    @pytest.fixture
    def classifier(self) -> LMSYSSilosClassifier:
        model_config = SiloClassifierConfig(
            pretrained_mpnet_name="sentence-transformers/all-mpnet-base-v2",
            num_labels=3,
        )
        return LMSYSSilosClassifier(model_config)

    @pytest.fixture
    def tokenizer(self, model_name: str) -> MPNetTokenizerFast:
        return AutoTokenizer.from_pretrained(model_name)

    def test(
        self,
        classifier: LMSYSSilosClassifier,
        tokenizer: MPNetTokenizerFast,
        datasets: Dataset,
    ) -> None:
        data_collator = Collator(
            tokenizer,
            max_length=256,
        )
        for i in range(len(datasets)):
            print(f">>>>>>>>>>>>>>>>>>>>> {i}")
            classifier(
                **data_collator(
                    {
                        "input_ids": datasets[i]["input_ids"],
                        "token_type_ids": datasets[i]["token_type_ids"],
                        "attention_mask": datasets[i]["attention_mask"],
                    }
                )
            )
