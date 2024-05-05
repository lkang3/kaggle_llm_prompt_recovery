import argparse
import yaml
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from utilities import get_device
from kaggle_llm_prompt_recovery.entities import DataConfig, TrainConfig, TokenizationConfig
from utilities import get_model_inputs
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

    # tokenizer
    tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_config.model_name, token=hf_token)

    # data setup
    data: pd.DataFrame = pd.read_csv(data_config.train_data_path)
    input_text = (
        "Generate a rewrite_prompt that effectively transforms the given original_text into the provided rewritten_text."
        "Capture the essence and context of the content while improving the language, coherence, and expressiveness."
        "Pay attention to detail, clarity, and overall quality in your generated rewrite_prompt."
        "Here is an example sample: original text-" + data.loc[0, 'original_text'] +
        "and this is the rewrite prompt for the bot-" + data.loc[0, 'rewrite_prompt'] +
        "and the expected rewritten text should be like-" + data.loc[0, 'rewritten_text'] +
        "Now, You will output in text the most suitable rewrite text. For the given original text- {ot}" + "and rewritten text-{rt}"+
        "then rewritten prompt- " +
        "<end_of_turn>"
    )
    device = get_device()
    tokenizer_params = {
        "truncation": True,
        "padding": "max_length",
        "max_length": 768,
        "return_tensors": "pt",
    }
    tokenized_inputs = tokenizer(input_text, **tokenizer_params)
    tokenized_inputs = get_model_inputs(device, tokenized_inputs)
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
    model.eval()
    outputs = model.generate(**tokenized_inputs, max_new_tokens=20)
    print(f">>>>>>>>>>>>>>>>>> {outputs}")  # P4T
    outputs = tokenizer.decode(outputs)
    print(f">>>>>>>>>>>>>>>>>> {outputs}")  # P4T


if __name__ == "__main__":
    # ! python /kaggle/working/go_only_inference.py --config_path=/kaggle/working/config.yaml --output_path /kaggle/working/output.csv
    hf_login(hf_token)
    go()
