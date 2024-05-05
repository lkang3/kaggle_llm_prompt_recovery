import logging
import sys
import yaml

import numpy as np
import pandas as pd

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from huggingface_hub import login as hf_login
from transformers import GemmaTokenizerFast
from transformers import GemmaForCausalLM
from transformers import Conversation


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def apply_gemma_chat_template_for_inference(
    original_text: str,
    rewrite_text: str,
    tokenizer: GemmaTokenizerFast,
    text_limit: int = 300,
) -> str:
    original_text: str = original_text[: text_limit]
    rewrite_text: str = rewrite_text[: text_limit]

    original_text: str = f"#TEXT#: {original_text}\n#END#"
    rewrite_text: str = f"#REWRITTEN TEXT#: {rewrite_text}\n#END#"
    question_text: str = (
        "What is the prompt to generate the REWRITTEN TEXT from the TEXT?\n"
    )

    conversation: Conversation = Conversation(
        messages=f"{original_text}{rewrite_text}{question_text}",
    )
    chat_text: str = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )

    return chat_text



SEED = 123
random_state = np.random.RandomState(SEED)
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)
numpy_gen = np.random.Generator(np.random.PCG64(seed=SEED))

hf_token: str = "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"
hf_login(hf_token)
device = get_device()
config_path = "/kaggle/working/config.yaml"
input_path = "/kaggle/input/llm-prompt-recovery/test.csv"
output_path = "/kaggle/working/"
model_path = "/kaggle/working/model_output/"

# load config file
with open(config_path, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

# tokenizer setup
tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(
    model_path,
    token=hf_token,
)

# test data setup (template + tokenization)
data_for_test: pd.DataFrame = pd.read_csv(input_path)
dataset_test: Dataset = Dataset.from_pandas(data_for_test)
dataset_test.cleanup_cache_files()
model_inputs = [
    apply_gemma_chat_template_for_inference(
        original_text=data["original_text"],
        rewrite_text=data["rewritten_text"],
        tokenizer=tokenizer,
    )
    for data in dataset_test
]
dataset_test = dataset_test.add_column("text", model_inputs)
cols_to_remove = set(["original_text", "rewritten_text"])
dataset_test = dataset_test.remove_columns(cols_to_remove)


# predict
model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer_params = {
    "truncation": True,
    "padding": "max_length",
    "max_length": 1024,
    "return_tensors": "pt",
    "add_special_tokens": False,
}
output_text = []
for data in dataset_test:
    inputs = tokenizer(data["text"], **tokenizer_params).to(device)
    raw_batch_outputs: torch.Tensor = model.generate(**inputs, max_new_tokens=50)

    mask: torch.Tensor = raw_batch_outputs != 0
    raw_batch_outputs: torch.Tensor = torch.masked_select(raw_batch_outputs, mask).reshape(1, -1)
    output = tokenizer.decode(raw_batch_outputs[0])
    start_of_model_turn: str = "<start_of_turn>model\n"
    if output.find(start_of_model_turn) < 0:
        output: str = "Please rewrite the original text."
    else:
        output: str = (
            output[output.find(start_of_model_turn) + len(start_of_model_turn):]
        ).strip()
        if output.find("<eos>") > 0:
            output: str = output[:output.find("<eos>")]
    output_text.append(output)
output: pd.DataFrame = pd.DataFrame(data={"rewrite_prompt": output_text})
output.to_csv(output_path, index=False)
