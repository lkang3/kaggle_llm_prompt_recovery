import logging
import sys
from pathlib import Path

import click
import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import login as hf_login
from peft import LoraConfig
from peft import get_peft_model
from train_utils import add_target
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import GemmaTokenizerFast
from transformers import TrainingArguments
from transformers import Trainer
from utils import clean_data
from utils import Collator
from utils import tokenization_separate
from utils import get_device

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
        data = data.iloc[:100, :]
    target_fields = [
        data_config["resp_a_win"],
        data_config["resp_b_win"],
        data_config["resp_tie"],
    ]
    add_target_field = data_config["added_target_field"]
    data = add_target(data, *target_fields, add_target_field)
    dataset = Dataset.from_pandas(data)
    dataset.cleanup_cache_files()
    tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(
        model_config["model_name"],
        token=hf_token,
    )
    tokenization_args = {
        "tokenizer": tokenizer,
        "resp_a_field": data_config["resp_a"],
        "resp_b_field": data_config["resp_b"],
        "max_token_length": 512,
        "max_resp_a_token_length": 255,
        "max_resp_b_token_length": 255,
        "target_field": add_target_field,
    }
    dataset = dataset.map(
        function=tokenization_separate,
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
        model_config["model_name"],
        num_labels=model_config["num_labels"],
    )
    lora_config = LoraConfig(
        r=8,
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, peft_config=lora_config)
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

    # save the entire peft model
    tokenizer.save_pretrained(model_output_dir)
    model = model.merge_and_unload()
    model.save_pretrained(model_output_dir)


if __name__ == "__main__":
    # ! python /kaggle/working/go_deberta.py --for_test=False
    hf_login(hf_token)
    go()
