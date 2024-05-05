import argparse
import logging
import sys
import yaml
import numpy as np
import pandas as pd
from typing import Dict

import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments

from kaggle_llm_prompt_recovery.utilities import get_device
from kaggle_llm_prompt_recovery.entities import DataConfig
from kaggle_llm_prompt_recovery.entities import TrainConfig
from kaggle_llm_prompt_recovery.entities import TokenizationConfig

from kaggle_llm_prompt_recovery.utils_train import get_max_train_steps
from kaggle_llm_prompt_recovery.utils_train import apply_gemma_chat_template_fo_training

from transformers import AdamW
from huggingface_hub import login as hf_login
from transformers import BitsAndBytesConfig
from transformers import GemmaTokenizerFast
from transformers import GemmaForCausalLM
from peft import get_peft_model
from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer


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


def go():

    device = get_device()

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    args = arg_parser.parse_args()
    config_path = args.config_path
    output_path = args.output_path

    # load config file
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    data_config: DataConfig = DataConfig.from_config(config["data"])
    tokenizer_config: TokenizationConfig = TokenizationConfig.from_config(config["tokenizer"])
    train_config: TrainConfig = TrainConfig.from_config(config["training"])

    # tokenizer setup
    tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(
        tokenizer_config.model_name,
        token=hf_token,
    )

    # train data setup (template)
    data_for_train: pd.DataFrame = pd.read_csv(data_config.train_data_path)
    data_for_train: Dataset = Dataset.from_pandas(data_for_train)
    data_for_train.cleanup_cache_files()
    response_token: str = "###PROMPT"
    model_inputs = [
        apply_gemma_chat_template_fo_training(
            original_text=data["original_text"],
            rewrite_text=data["rewritten_text"],
            prompt_text=data["rewrite_prompt"],
            tokenizer=tokenizer,
            response_token=response_token,
        )
        for data in data_for_train
    ]
    data_for_train_0 = data_for_train.add_column("text", model_inputs)
    model_inputs = [
        apply_gemma_chat_template_fo_training(
            original_text=data["original_text"],
            rewrite_text=data["rewritten_text"],
            prompt_text=data["rewrite_prompt"],
            tokenizer=tokenizer,
            response_token=response_token,
            reverse_select=True,
        )
        for data in data_for_train
    ]
    data_for_train_1 = data_for_train.add_column("text", model_inputs)
    data_for_train = concatenate_datasets([data_for_train_0, data_for_train_1])
    cols_to_remove = set(["original_text", "rewritten_text", "rewrite_prompt"])
    data_for_train = data_for_train.remove_columns(cols_to_remove)
    data_for_train: DatasetDict = data_for_train.train_test_split(
        test_size=0.2, seed=SEED, generator=numpy_gen,
    )
    dataset_train: Dataset = data_for_train["train"]
    dataset_eval: Dataset = data_for_train["test"]

    import pdb  # P4T
    pdb.set_trace()  # P4T

    # training
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
        tokenizer_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = get_peft_model(model, peft_config=lora_config)
    torch.cuda.empty_cache()

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_token,
        mlm=False,
    )
    train_size = dataset_train.shape[0]
    train_batch_size: int = train_config.train_batch_size
    model_output_dir: str = "model_output"
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        dataset_text_field="text",
        peft_config=lora_config,
        args=TrainingArguments(
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=train_config.num_gradient_update_batch,
            num_train_epochs=train_config.num_epoch,
            warmup_steps=0.03,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir=model_output_dir,
            optim="paged_adamw_8bit",
            save_strategy="epoch",
            report_to="none",
            max_steps=train_config.num_epoch * get_max_train_steps(train_size, train_batch_size),
            evaluation_strategy="epoch",
            do_eval=True,
        ),
        max_seq_length=tokenizer_config.max_length,
        data_collator=data_collator,
    )
    trainer.train()

    # save the entire peft model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(model_output_dir, safe_serialization=True, max_shard_size="2GB")


def go_cv():
    device = get_device()

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)
    args = arg_parser.parse_args()
    config_path = args.config_path
    output_path = args.output_path

    # load config file
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    data_config: DataConfig = DataConfig.from_config(config["data"])
    tokenizer_config: TokenizationConfig = TokenizationConfig.from_config(config["tokenizer"])
    train_config: TrainConfig = TrainConfig.from_config(config["training"])

    # tokenizer setup
    tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(
        tokenizer_config.model_name,
        token=hf_token,
    )

    # train data setup (template)
    data_for_train: pd.DataFrame = pd.read_csv(data_config.train_data_path)
    data_for_train: Dataset = Dataset.from_pandas(data_for_train)
    data_for_train.cleanup_cache_files()
    response_token: str = "###PROMPT###"
    model_inputs = [
        apply_gemma_chat_template_fo_training(
            original_text=data["original_text"],
            rewrite_text=data["rewritten_text"],
            prompt_text=data["rewrite_prompt"],
            tokenizer=tokenizer,
            response_token=response_token,
        )
        for data in data_for_train
    ]
    data_for_train_0 = data_for_train.add_column("text", model_inputs)
    model_inputs = [
        apply_gemma_chat_template_fo_training(
            original_text=data["original_text"],
            rewrite_text=data["rewritten_text"],
            prompt_text=data["rewrite_prompt"],
            tokenizer=tokenizer,
            response_token=response_token,
            reverse_select=True,
        )
        for data in data_for_train
    ]
    data_for_train_1 = data_for_train.add_column("text", model_inputs)
    data_for_train = concatenate_datasets([data_for_train_0, data_for_train_1])
    cols_to_remove = set(["original_text", "rewritten_text", "rewrite_prompt"])
    data_for_train = data_for_train.remove_columns(cols_to_remove)

    num_of_cv: int = train_config.partition_cv.folds
    best_score: float = None
    for _ in range(num_of_cv):
        data_for_train: DatasetDict = data_for_train.train_test_split(
            test_size=0.2, seed=SEED, generator=numpy_gen,
        )
        dataset_train: Dataset = data_for_train["train"]
        dataset_eval: Dataset = data_for_train["test"]

        # training
        lora_config = LoraConfig(
            r=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
            tokenizer_config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = get_peft_model(model, peft_config=lora_config)
        torch.cuda.empty_cache()

        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_token,
            mlm=False,
        )
        train_size = dataset_train.shape[0]
        train_batch_size: int = train_config.train_batch_size
        model_output_dir: str = "model_output"
        trainer: SFTTrainer = SFTTrainer(
            model=model,
            train_dataset=dataset_train,
            eval_dataset=dataset_eval,
            dataset_text_field="text",
            peft_config=lora_config,
            args=TrainingArguments(
                per_device_train_batch_size=train_batch_size,
                gradient_accumulation_steps=train_config.num_gradient_update_batch,
                num_train_epochs=train_config.num_epoch,
                warmup_steps=0.03,
                learning_rate=2e-4,
                logging_steps=1,
                output_dir=model_output_dir,
                optim="paged_adamw_8bit",
                save_strategy="no",
                report_to="none",
                max_steps=train_config.num_epoch * get_max_train_steps(train_size, train_batch_size),
            ),
            max_seq_length=tokenizer_config.max_length,
            data_collator=data_collator,
        )
        trainer.train()
        eval_scores: Dict[str, float] = trainer.evaluate(dataset_eval)
        print(f">>>>>>>>>>>>>>>>>>>>>>> {eval_scores}")

    # save the entire peft model
    model.save_pretrained(model_output_dir)
    merged_model = model.merge_and_unload()
    model.save_pretrained(model_output_dir)
    merged_model.save_pretrained(model_output_dir, safe_serialization=True, max_shard_size="2GB")

    allocator = torch.cuda.memory._get_current_allocator()
    allocator.set_max_split_size()


if __name__ == "__main__":
    # ! python /kaggle/working/go.py --config_path=/kaggle/working/config.yaml --output_path /kaggle/working/submission.csv
    hf_login(hf_token)
    go()
