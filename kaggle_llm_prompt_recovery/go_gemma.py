import argparse
import yaml
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

import torch
from datasets import Dataset
from datasets import DatasetDict
from accelerate import Accelerator
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from utilities import get_device
from utilities import preprocess_tokenization
from kaggle_llm_prompt_recovery.utils_train import trainer
from kaggle_llm_prompt_recovery.entities import DataConfig, TrainConfig, TokenizationConfig
from utilities import DataCollatorHandler
from kaggle_llm_prompt_recovery.utils_inference import run_gemma_inference, predict
from utilities import get_data_loader
from utilities import run_loss_func
from torch.optim import lr_scheduler
from torch.nn import CosineSimilarity
from transformers import AdamW
from huggingface_hub import login as hf_login
from transformers import BitsAndBytesConfig
from transformers import GemmaTokenizerFast
from transformers import GemmaForCausalLM
from peft import get_peft_model
from peft import LoraConfig


SEED = 123
random_state = np.random.RandomState(SEED)
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)

hf_token: str = "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"


def go():
    # https://huggingface.co/blog/gemma-peft
    # https://huggingface.co/blog/4bit-transformers-bitsandbytes

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
    tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_config.model_name, token=hf_token)

    # data setup
    data: pd.DataFrame = pd.read_csv(data_config.train_data_path)
    data = data.loc[: 50, :]  # P4T
    dataset: Dataset = Dataset.from_pandas(data)
    dataset.cleanup_cache_files()
    ds_preprocess_params = {
        "tokenizer": tokenizer,
        "max_length": tokenizer_config.max_length,
        "original_txt_field_name": "original_text",
        "rewrite_txt_field_name": "rewritten_text",
        "target_field_name": "rewrite_prompt",
    }
    data_features_to_remove = list(data.columns)
    dataset: Dataset = dataset.map(
        preprocess_tokenization,
        fn_kwargs=ds_preprocess_params,
        batched=True,
        batch_size=20,
        remove_columns=data_features_to_remove,
    )
    dataset: DatasetDict = dataset.train_test_split(test_size=0.2, seed=SEED)
    dataset_for_train: DatasetDict = dataset["train"].train_test_split(test_size=0.2, seed=SEED)
    dataset_train: Dataset = dataset_for_train["train"]
    dataset_eval: Dataset = dataset_for_train["test"]
    target_name = "labels"
    dataset_holdout: Dataset = dataset["test"].remove_columns(target_name)

    data_loader_params = {
        "collate_fn": DataCollatorHandler(),
        "batch_size": train_config.train_batch_size,
        "generator": torch_gen,
        "shuffle": True,
    }
    data_train_loader = get_data_loader(dataset=dataset_train, **data_loader_params)
    data_eval_loader = get_data_loader(dataset=dataset_eval, **data_loader_params)
    data_holdout_loader = get_data_loader(dataset=dataset_holdout, **data_loader_params)

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
    device = get_device()
    loss_func = run_loss_func(CosineSimilarity, device)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate)
    learning_rate_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=600,
        eta_min=train_config.min_learning_rate,
    )
    accelerator = Accelerator()
    trainer(
        device=device,
        model=model,
        tokenizer=tokenizer,
        train_data_loader=data_train_loader,
        eval_data_loader=data_eval_loader,
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        num_epoch=train_config.num_epoch,
        num_gradient_update_batch=train_config.num_gradient_update_batch,
        run_model_inference=run_gemma_inference,
        loss_func=loss_func,
        accelerator=accelerator,
    )

    # predict
    token_outputs: List[torch.Tensor] = predict(
        device=device,
        model=model,
        data_loader=data_holdout_loader,
        run_model_inference=run_gemma_inference,
    )
    decoded_outputs: List[str] = tokenizer.batch_decode(token_outputs)

    output: pd.DataFrame = pd.DataFrame(data=decoded_outputs, columns=["prompt"])
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    # ! python /kaggle/working/go_only_inference.py --config_path=/kaggle/working/config.yaml --output_path /kaggle/working/output.csv
    hf_login(hf_token)
    go()