import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer
from transformers import AutoModel

from train_utils import add_target
from utils import clean_data
from utils import get_device
from utils import run_embedding_feature_engineering


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


SEED = 123
random_state = np.random.RandomState(SEED)
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)
numpy_gen = np.random.Generator(np.random.PCG64(seed=SEED))


hf_token: str = "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"


@click.command()
@click.option("--for_test", type=bool, default=False, required=True)
def go(for_test: bool) -> None:
    device = get_device()

    data_config = {
        "train_data_path": "/home/lkang/Downloads/lmsys-chatbot-arena/train.csv",
        "test_data_path": "/home/lkang/Downloads/lmsys-chatbot-arena/test.csv",
        "prompt": "prompt",
        "resp_a": "response_a",
        "resp_b": "response_b",
        "resp_a_win": "winner_model_a",
        "resp_b_win": "winner_model_b",
        "resp_tie": "winner_tie",
        "added_target_field": "labels",
    }
    model_config = {
        "model_name": "microsoft/deberta-base",
        "num_labels": 3,
        "train_pct": 0.8,
        "eval_pct": 0.2,
        "cv": 4,
        "max_length": 256,
        "learning_rate": 7.0e-06,
        "train_batch_size": 4,
        "num_epoch": 1,
        "output_path": "model_output",
    }

    data_path = Path(data_config["train_data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(data_path, input_fields)
    if for_test:
        data = data.iloc[:10, :]

    target_fields = [
        data_config["resp_a_win"],
        data_config["resp_b_win"],
        data_config["resp_tie"],
    ]
    add_target_field = data_config["added_target_field"]
    data = add_target(data, *target_fields, add_target_field)

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"], token=hf_token)
    llm_model = AutoModel.from_pretrained(model_config["model_name"], token=hf_token)
    data_embeddings = run_embedding_feature_engineering(
        data=data,
        tokenizer=tokenizer,
        prompt_field=data_config["prompt"],
        resp_a_field=data_config["resp_a"],
        resp_b_field=data_config["resp_b"],
        max_prompt_token_length=64,
        max_resp_a_token_length=255,
        max_resp_b_token_length=255,
        llm_model=llm_model,
    )


if __name__ == "__main__":
    # ! python /kaggle/working/go_lgbm.py --for_test=False
    hf_login(hf_token)
    go()
