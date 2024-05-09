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
from train_utils import get_max_train_steps
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import DataCollatorWithPadding
from transformers import GemmaTokenizerFast
from transformers import TrainingArguments
from trl import SFTTrainer
from utils import clean_data
from utils import tokenization_separate

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
@click.option("--for_test", type=bool, default=True, required=True)
def go(for_test: bool) -> None:
    data_config = {
        "train_data_path": "/kaggle/input/lmsys-chatbot-arena/train.csv",
        "test_data_path": "/kaggle/input/lmsys-chatbot-arena/test.csv",
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
        "max_length": 256,
        "learning_rate": 7.0e-06,
        "train_batch_size": 4,
        "num_gradient_update_batch": 4,
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
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="SEQ_CLS",
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["model_name"],
        num_labels=model_config["num_labels"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    peft_model = get_peft_model(model, peft_config=lora_config)
    torch.cuda.empty_cache()
    data_collator = DataCollatorWithPadding(
        tokenizer,
        max_length=model_config["max_length"],
    )
    train_size = dataset_train.shape[0]
    train_batch_size: int = model_config["train_batch_size"]
    model_output_dir: str = model_config["output_path"]
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        dataset_text_field="text",
        peft_config=lora_config,
        args=TrainingArguments(
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=model_config["num_gradient_update_batch"],
            num_train_epochs=model_config["num_epoch"],
            warmup_steps=0.03,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir=model_output_dir,
            optim="paged_adamw_8bit",
            save_strategy="epoch",
            report_to="none",
            max_steps=model_config["num_epoch"]
            * get_max_train_steps(train_size, train_batch_size),
        ),
        max_seq_length=model_config["max_length"],
        data_collator=data_collator,
    )
    trainer.train()

    # save the entire peft model
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(
        model_output_dir, safe_serialization=True, max_shard_size="2GB"
    )


if __name__ == "__main__":
    # ! python /kaggle/working/go.py --for_test=True
    hf_login(hf_token)
    go()
