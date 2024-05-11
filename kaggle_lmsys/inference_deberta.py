from pathlib import Path

import click
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

from utils import clean_data
from utils import tokenization
from utils import get_device


SEED = 123
random_state = np.random.RandomState(SEED)
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)
numpy_gen = np.random.Generator(np.random.PCG64(seed=SEED))
device = get_device()


@click.command()
@click.option('--for_test', type=bool, default=False, required=True)
def go(for_test: bool):
    data_config = {
        "train_data_path": "/kaggle/input/lmsys-chatbot-arena/train.csv",
        "prompt": "prompt",
        "resp_a": "response_a",
        "resp_b": "response_b",
        "resp_a_win": "winner_model_a",
        "resp_b_win": "winner_model_b",
        "resp_tie": "winner_tie",
        "added_target_field": "label",
    }
    model_config = {
        "model_name": "google/gemma-2b-it",
        "num_labels": 3,
        "train_pct": 0.8,
        "eval_pct": 0.2,
        "cv": 4,
        "max_length": 512,
        "learning_rate": 7.0e-06,
        "train_batch_size": 4,
        "num_gradient_update_batch": 4,
        "num_epoch": 1,
        "output_path": "model_output",

    }
    inference_config = {
        "data_path": "/kaggle/input/lmsys-chatbot-arena/test.csv",
        "tokenizer_path": "/kaggle/input/lmsys-model/model_output/checkpoint-2",
        "model_path": "/kaggle/input/lmsys-model/model_output/",
        "output_path": "/kaggle/working/submission.csv",
    }
    tokenizer = AutoTokenizer.from_pretrained(
        inference_config["tokenizer_path"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        inference_config["model_path"]
    )
    model.to(device)

    data_path = Path(inference_config["data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(data_path, input_fields)
    if for_test:
        data = data.iloc[: 10, :]
    dataset = Dataset.from_pandas(data)
    dataset.cleanup_cache_files()
    tokenization_args = {
        "tokenizer": tokenizer,
        "max_length": model_config["max_length"],
        "prompt_field": data_config["prompt"],
        "resp_a_field": data_config["resp_a"],
        "resp_b_field": data_config["resp_b"],
        "target_field": data_config["added_target_field"],
    }
    dataset_id = np.array(dataset["id"]).reshape((len(dataset), -1))
    dataset = dataset.map(
        function=tokenization,
        batched=False,
        fn_kwargs=tokenization_args,
        remove_columns=dataset.column_names,
    )
    data_collator = DataCollatorWithPadding(
        tokenizer,
        max_length=model_config["max_length"],
    )

    preds_data = np.zeros((len(dataset), model_config["num_labels"]))
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
    output.to_csv(inference_config["output_path"], index=False)


if __name__ == "__main__":
    # ! python /kaggle/working/go.py --for_test=True
    go()
