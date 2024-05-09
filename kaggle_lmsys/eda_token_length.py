import logging
import sys
from pathlib import Path

import click
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import GemmaTokenizerFast
from huggingface_hub import login as hf_login

from utils import clean_data
from utils import get_tokenization_length


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
@click.option('--for_test', type=bool, default=True, required=True)
@click.option('--input_data', type=str, required=True)
@click.option('--output_data', type=str, required=True)
def go(
    for_test: bool,
    input_data: str,
    output_data: str,
) -> None:
    data_config = {
        "train_data_path": "",
        "prompt": "prompt",
        "resp_a": "response_a",
        "resp_b": "response_b",
    }
    model_config = {
        "model_name": "google/gemma-2b-it",  # "google-bert/bert-base-uncased"
        "max_length": 1024,

    }

    data_path = Path(input_data)
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(data_path, input_fields)
    if for_test:
        data = data.iloc[: 100, :]
    tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(
        model_config["model_name"],
        token=hf_token,
    )
    func_args = {
        "tokenizer": tokenizer,
        "max_length": model_config["max_length"],
    }
    for input_field in input_fields:
        data[f"{input_field}_token_length"] = data.loc[:, [input_field]].map(
            get_tokenization_length,
            **func_args,
        ).values.flatten()
    data.to_csv(Path(output_data), index=False)


if __name__ == "__main__":
    """
    ! python go.py --for_test=True --input_data='/home/lkang/Downloads/lmsys-chatbot-arena/train.csv' --output_data='/home/lkang/Downloads/lmsys-chatbot-arena/train_eda.csv'
    """

    """
    import pandas as pd
    df = pd.read_csv('/kaggle/working/train_eda.csv')
    df.loc[:, ["prompt_token_length", "response_a_token_length", "response_b_token_length"]].hist(bins=100)
    prompt_token_length = df["prompt_token_length"]
    response_a_token_length = df["response_a_token_length"]
    response_b_token_length = df["response_b_token_length"]
    print(prompt_token_length.quantile(0.5), prompt_token_length.quantile(0.75))
    print(response_a_token_length.quantile(0.5), response_a_token_length.quantile(0.75))
    print(response_b_token_length.quantile(0.5), response_b_token_length.quantile(0.75))
    """

    hf_login(hf_token)
    go()