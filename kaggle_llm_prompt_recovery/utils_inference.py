from typing import Callable, List

import torch
from torch import autocast
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import GemmaTokenizerFast, Conversation, GemmaForCausalLM

from kaggle_llm_prompt_recovery.utilities import time_it, inference_wrapper


def apply_gemma_chat_template_for_inference(
    original_text: str,
    rewrite_text: str,
    tokenizer: GemmaTokenizerFast,
    text_limit: int = 250,
) -> str:
    original_text: str = original_text[: text_limit]
    rewrite_text: str = rewrite_text[: text_limit]

    original_text: str = f"The original text is: {original_text}\n"
    rewrite_text: str = f"The rewritten text is: {rewrite_text}\n"
    question_text: str = (
        "What is the prompt used to generate the rewritten text from the original text?\n"
    )

    conversation: Conversation = Conversation(
        messages=f"{original_text}{rewrite_text}{question_text}",
    )
    chat_text: str = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )

    return chat_text


@time_it
def run_gemma_inference(
    model: GemmaForCausalLM,
    **inputs,
) -> torch.Tensor:
    """This function encapsulates
    - Logic to call a model
    - Logic to post process model outputs

    :param model:
    :return:
    """
    model_generate_outputs: torch.tensor = model.generate(**inputs, max_new_tokens=50)
    return model_generate_outputs


@torch.no_grad()
@inference_wrapper
def predict(
    device: torch.device,
    model: Module,
    tokenizer: object,
    data_loader: DataLoader,
    run_model_inference: Callable,
) -> List[str]:
    outputs: List[str] = []

    start_of_model_turn: str = "<start_of_turn>model\n"
    with autocast(device_type=device.type):
        for batch in data_loader:
            inputs = batch
            raw_batch_outputs: torch.Tensor = run_model_inference(model=model, **inputs)
            mask: torch.Tensor = raw_batch_outputs != -100
            raw_batch_outputs: torch.Tensor = torch.masked_select(
                raw_batch_outputs,
                mask,
            ).reshape(raw_batch_outputs.shape)
            raw_batch_outputs: List[str] = tokenizer.batch_decode(raw_batch_outputs)
            batch_outputs: List[str] = []
            print(f">>>>>>>>>>>>>>>>>22222222222222 {raw_batch_outputs}")
            for batch_output in raw_batch_outputs:
                output: str = (
                    batch_output[batch_output.find(start_of_model_turn) + len(start_of_model_turn):]
                ).strip()
                output = output or "n/a"
                batch_outputs.append(output)
            outputs.extend(batch_outputs)
    return outputs
