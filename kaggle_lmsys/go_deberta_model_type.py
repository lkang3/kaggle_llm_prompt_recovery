import logging
import sys
from pathlib import Path

import click
import numpy as np
from peft.peft_model import PeftModel
import torch
from datasets import Dataset
from huggingface_hub import login as hf_login
from kaggle_lmsys.train_utils import create_model_type_train_data
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import GemmaTokenizerFast
from transformers import TrainingArguments
from transformers import Trainer
from kaggle_lmsys.utils import clean_data
from kaggle_lmsys.utils import Collator
from kaggle_lmsys.utils import tokenization_prompt_one_resp
from kaggle_lmsys.utils import get_device

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
        "train_data_path": "/kaggle/input/lmsys-chatbot-arena/train.csv",
        "test_data_path": "/kaggle/input/lmsys-chatbot-arena/test.csv",
        "prompt": "prompt",
        "resp_a": "response_a",
        "resp_b": "response_b",
    }
    model_config = {
        "model_name": "microsoft/deberta-base",
        "train_pct": 0.8,
        "eval_pct": 0.2,
        "cv": 4,
        "max_length": 256,
        "learning_rate": 7.0e-06,
        "train_batch_size": 4,
        "num_epoch": 1,
        "output_path": "model_type_classifier_output",
    }

    data_path = Path(data_config["train_data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(data_path, input_fields)
    if for_test:
        data = data.iloc[:100, :]
    train_data_config = {
        "prompt_field_name": data_config["prompt"],
        "response_field_name": "response",
        "model_type_field_name": "labels",
    }
    data, num_of_targets = create_model_type_train_data(
        data,
        output_prompt_field_name=train_data_config["prompt_field_name"],
        output_response_field_name=train_data_config["response_field_name"],
        output_model_type_field_name=train_data_config["model_type_field_name"],
    )
    dataset = Dataset.from_pandas(data)
    dataset.cleanup_cache_files()
    tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(
        model_config["model_name"],
        token=hf_token,
    )
    tokenization_args = {
        "tokenizer": tokenizer,
        "max_length": 512,
        "prompt_field": data_config["prompt"],
        "resp_field": train_data_config["response_field_name"],
        "target_field": train_data_config["model_type_field_name"],
    }
    dataset = dataset.map(
        function=tokenization_prompt_one_resp,
        batched=False,
        fn_kwargs=tokenization_args,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.train_test_split(
        test_size=model_config["eval_pct"],
        seed=SEED,
        generator=numpy_gen,
    )
    dataset_train: Dataset = dataset["train"]
    dataset_eval: Dataset = dataset["test"]

    # modeling setup
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-base",
        num_labels=num_of_targets,
    )
    torch.cuda.empty_cache()
    data_collator = Collator(
        tokenizer,
        max_length=model_config["max_length"],
    )
    train_batch_size: int = model_config["train_batch_size"]
    model_output_dir: str = model_config["output_path"]
    trainer = Trainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        args=TrainingArguments(
            per_device_train_batch_size=train_batch_size,
            num_train_epochs=model_config["num_epoch"],
            warmup_steps=0.03,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir=model_output_dir,
            optim="paged_adamw_8bit",
            save_strategy="epoch",
            report_to="none",
            dataloader_pin_memory=False,
        ),
        data_collator=data_collator,
    )
    trainer.train()

    tokenizer.save_pretrained(model_output_dir)
    if isinstance(model, PeftModel):
        # save the entire adapter model
        model = model.merge_and_unload()
    model.save_pretrained(model_output_dir)


if __name__ == "__main__":
    # ! python /kaggle/working/go_deberta.py --for_test=False
    hf_login(hf_token)
    go()
