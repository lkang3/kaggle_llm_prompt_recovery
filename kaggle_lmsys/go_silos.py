import logging
import sys
from pathlib import Path

import click
import numpy as np
from peft.peft_model import PeftModel
import torch
import yaml
from datasets import Dataset
from huggingface_hub import login as hf_login
from kaggle_lmsys.train_utils import add_target
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from kaggle_lmsys.utils import clean_data
from kaggle_lmsys.utils import Collator
from kaggle_lmsys.utils import tokenization_separate
from kaggle_lmsys.models.silos_classifier import LMSYSSilosClassifier
from kaggle_lmsys.models.silos_classifier import SiloClassifierConfig

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


SEED = 123
random_state = np.random.RandomState(SEED)
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)
numpy_gen = np.random.Generator(np.random.PCG64(seed=SEED))


@click.command()
@click.option("--hf_token", type=str, required=False)
@click.option("--for_test", type=bool, default=False, required=True)
@click.option("--for_test_pct", type=float, default=1.0, required=True)
@click.option("--config_path", type=str, required=True)
def go(
    hf_token: str,
    for_test: bool,
    for_test_pct: float,
    config_path: str,
) -> None:
    # login
    hf_login(hf_token)
    with open(Path(config_path), "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]
    training_config = config["silo_classifier"]["train"]
    tokenizer_config = config["silo_classifier"]["tokenizer"]
    classifier_config = config["silo_classifier"]["classifier"]
    tokenizer_name = tokenizer_config["name"]

    data_path = Path(data_config["train_data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(data_path, input_fields)
    if for_test:
        num_samples = int(for_test_pct * len(data))
        data = data.iloc[:num_samples, :]

    target_fields = [
        data_config["resp_a_win"],
        data_config["resp_b_win"],
        data_config["resp_tie"],
    ]
    add_target_field = data_config["added_target_field"]
    data = add_target(data, *target_fields, add_target_field)
    dataset = Dataset.from_pandas(data)
    dataset.cleanup_cache_files()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        token=hf_token,
    )
    tokenization_args = {
        "tokenizer": tokenizer,
        "prompt_field": data_config["prompt"],
        "resp_a_field": data_config["resp_a"],
        "resp_b_field": data_config["resp_b"],
        "max_token_length": classifier_config["max_length"],
        "max_prompt_token_length": tokenizer_config["max_prompt_token_length"],
        "max_resp_a_token_length": tokenizer_config["max_resp_a_token_length"],
        "max_resp_b_token_length": tokenizer_config["max_resp_b_token_length"],
        "target_field": add_target_field,
    }
    dataset = dataset.map(
        function=tokenization_separate,
        batched=False,
        fn_kwargs=tokenization_args,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.train_test_split(
        test_size=training_config["eval_pct"],
        seed=SEED,
        generator=numpy_gen,
    )
    dataset_train: Dataset = dataset["train"]
    dataset_eval: Dataset = dataset["test"]

    # modeling setup
    model_config = SiloClassifierConfig(
        pretrained_mpnet_name=classifier_config["name"],
        num_labels=classifier_config["num_labels"],
    )
    model = LMSYSSilosClassifier(model_config)
    model.update_silos_mpnet_with_pretrained_model(model_config.pretrained_mpnet_name)
    torch.cuda.empty_cache()
    data_collator = Collator(
        tokenizer,
        max_length=classifier_config["max_length"],
    )
    train_batch_size: int = training_config["train_batch_size"]
    model_output_dir: str = "model_output"
    trainer = Trainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        args=TrainingArguments(
            per_device_train_batch_size=train_batch_size,
            num_train_epochs=training_config["num_epoch"],
            warmup_steps=0.03,
            learning_rate=training_config["learning_rate"],
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
    # python ./kaggle_lmsys/go_silos.py --hf_token=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 --for_test=True --for_test_pct=0.002 --config_path=./kaggle_lmsys/config/all_in_one.yaml
    go()
