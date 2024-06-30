from pathlib import Path

import click
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import yaml

from utils import clean_data
from utils import tokenization_separate
from utils import get_device
from utils import Collator
from kaggle_lmsys.models.silos_classifier import LMSYSSilosClassifier


SEED = 123
random_state = np.random.RandomState(SEED)
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)
numpy_gen = np.random.Generator(np.random.PCG64(seed=SEED))
device = get_device()


@torch.no_grad()
@click.command()
@click.option("--for_test", type=bool, default=False, required=True)
@click.option("--for_test_pct", type=float, default=1.0, required=True)
@click.option("--config_path", type=str, required=True)
def go(
    for_test: bool,
    for_test_pct: float,
    config_path: str,
):

    with open(Path(config_path), "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]
    tokenizer_config = config["silo_classifier"]["tokenizer"]
    classifier_config = config["silo_classifier"]["classifier"]
    model_path = "/kaggle/working/model_output"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LMSYSSilosClassifier.from_pretrained(model_path, ignore_mismatched_sizes=True)
    model.to(device)

    data_path = Path(data_config["test_data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(data_path, input_fields)
    if for_test:
        num_samples = int(for_test_pct * len(data))
        data = data.iloc[:num_samples, :]
    dataset_id = np.array(data["id"]).reshape((len(data), -1))

    dataset = Dataset.from_pandas(data)
    dataset.cleanup_cache_files()
    tokenization_args = {
        "tokenizer": tokenizer,
        "prompt_field": data_config["prompt"],
        "resp_a_field": data_config["resp_a"],
        "resp_b_field": data_config["resp_b"],
        "max_token_length": classifier_config["max_length"],
        "max_prompt_token_length": tokenizer_config["max_prompt_token_length"],
        "max_resp_a_token_length": tokenizer_config["max_resp_a_token_length"],
        "max_resp_b_token_length": tokenizer_config["max_resp_b_token_length"],
        "target_field": None,
    }
    dataset = dataset.map(
        function=tokenization_separate,
        batched=False,
        fn_kwargs=tokenization_args,
        remove_columns=dataset.column_names,
    )
    data_collator = Collator(
        tokenizer,
        max_length=classifier_config["max_length"],
    )

    preds_data = np.zeros((len(dataset), classifier_config["num_labels"]))
    for row_id, data in enumerate(dataset):
        preds = model(**data_collator([data]))
        preds = torch.nn.functional.softmax(preds.logits.cuda(), dim=-1)
        preds = preds.detach().cpu().numpy().flatten()
        preds_data[row_id] = preds
    data = np.concatenate((dataset_id, preds_data), axis=1)
    output: pd.DataFrame = pd.DataFrame(
        data=data,
        columns=["id", "winner_model_a", "winner_model_b", "winner_tie"],
    )
    output.to_csv("/kaggle/working/submission.csv", index=False)


if __name__ == "__main__":
    # ! python /kaggle/working/inference_go_silos.py --for_test=True --for_test_pct=0.002 --config_path=./kaggle_lmsys/config/all_in_one.yaml
    go()
