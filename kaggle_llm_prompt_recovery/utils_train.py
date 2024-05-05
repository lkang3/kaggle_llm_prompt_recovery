import gc
import logging
import math
import sys
from typing import Dict, Callable, Optional, List, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GemmaTokenizerFast, Conversation, GemmaForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import SFTTrainer

from kaggle_llm_prompt_recovery.utilities import time_it
from kaggle_llm_prompt_recovery.utilities import get_model_inputs
from kaggle_llm_prompt_recovery.utilities import inference_wrapper
from kaggle_llm_prompt_recovery.utilities import get_model_targets


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def get_max_train_steps(num_of_samples: int, train_batch_size: int) -> int:
    return math.ceil(num_of_samples / train_batch_size)


def apply_phi_chat_template_for_training(
    original_text: str,
    rewrite_text: str,
    prompt_text: str,
    tokenizer: GemmaTokenizerFast,
    response_token: str = "###PROMPT###",
    text_limit: int = 250,
    reverse_select: bool = False,
) -> str:
    if reverse_select:
        original_text: str = original_text[-text_limit:]
        rewrite_text: str = rewrite_text[-text_limit:]
    else:
        original_text: str = original_text[: text_limit]
        rewrite_text: str = rewrite_text[: text_limit]
    prompt_text: str = prompt_text
    original_text: str = f"Instruct: Original Text: {original_text}\n"
    rewrite_text: str = f"Rewritten Text:: {rewrite_text}\n"
    question_text: str = (
        "Write a prompt that was likely given to the LLM to rewrite original text into rewritten text.\n"
    )
    prompt_text: str = f"{response_token}{prompt_text}"
    conversation: Conversation = Conversation(
        messages=f"{original_text}{rewrite_text}{question_text}",
    )
    chat_text: str = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )

    return f"{chat_text}{prompt_text}"


def apply_gemma_chat_template_fo_training(
    original_text: str,
    rewrite_text: str,
    prompt_text: str,
    tokenizer: GemmaTokenizerFast,
    response_token: str = "###PROMPT###",
    text_limit: int = 250,
    reverse_select: bool = False,
) -> str:
    if reverse_select:
        original_text: str = original_text[-text_limit:]
        rewrite_text: str = rewrite_text[-text_limit:]
    else:
        original_text: str = original_text[: text_limit]
        rewrite_text: str = rewrite_text[: text_limit]
    prompt_text: str = prompt_text
    original_text: str = f"The original text is: {original_text}\n"
    rewrite_text: str = f"The rewritten text is: {rewrite_text}\n"
    question_text: str = (
        "What is the prompt used to generate the rewritten text from the original text?\n"
    )

    prompt_text: str = f"{response_token}{prompt_text}"
    conversation: Conversation = Conversation(
        messages=f"{original_text}{rewrite_text}{question_text}",
    )
    chat_text: str = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )

    return f"{chat_text}{prompt_text}"


class MyTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs: bool = False, ignored_token: int = -100):
        has_label: bool = "labels" in inputs
        if has_label:
            outputs = model(**inputs)
            labels = inputs.pop("labels")
            normalized_outputs = torch.softmax(outputs.logits, dim=-1)
            outputs: torch.Tensor = torch.max(normalized_outputs, dim=-1).indices
            mask_of_ignore: torch.tensor = labels != ignored_token
            outputs: torch.tensor = torch.masked_select(outputs, mask_of_ignore).reshape(1, -1)
            labels: torch.tensor = torch.masked_select(labels, mask_of_ignore).reshape(1, -1)
            return sharpened_consine_similarity_loss(outputs, labels)
        else:
            return super().compute_loss(model, inputs, return_outputs)


@time_it
def run_gemma_forward(
    model: GemmaForCausalLM,
    inputs: Dict[str, torch.Tensor],
) -> CausalLMOutputWithPast:
    outputs: CausalLMOutputWithPast = model(**inputs)
    return outputs


def train_one_epoch(
    device: torch.device,
    model: Module,
    train_data_loader: DataLoader,
    optimizer: object,
    learning_rate_scheduler: LRScheduler,
    num_gradient_update_batch: int,
    run_model_inference: Callable,
    loss_func: Callable,
    accelerator: Optional[Accelerator] = None,
    enable_grad_scale: Optional[bool] = False,
) -> float:
    to_accelerate = accelerator is not None
    scaler = GradScaler(enabled=enable_grad_scale)
    batch_loss: List[float] = []
    num_of_batch = len(train_data_loader)

    for batch_idx, batch in enumerate(train_data_loader):
        inputs: Dict[str, torch.Tensor] = get_model_inputs(device, batch, is_inference=False)
        outputs: CausalLMOutputWithPast = run_gemma_forward(model, inputs)

        loss: torch.FloatTensor = outputs.loss
        batch_loss.append(loss.item())
        # accumulate gradients (scaled)
        if to_accelerate:
            accelerator.backward(scaler.scale(loss))
        else:
            scaler.scale(loss).backward()
        # weight update
        if (
            batch_idx + 1
        ) % num_gradient_update_batch == 0 or batch_idx == num_of_batch - 1:
            # unscale gradients
            scaler.step(optimizer)
            # update weights
            scaler.update()
            # reset the gradients
            optimizer.zero_grad()
            # adjust learning rate
            learning_rate_scheduler.step()

    torch.cuda.empty_cache()
    gc.collect()

    return np.mean(batch_loss)


def trainer(
    device: torch.device,
    model: Module,
    tokenizer: GemmaTokenizerFast,
    train_data_loader: DataLoader,
    eval_data_loader: DataLoader,
    optimizer: Optimizer,
    learning_rate_scheduler: LRScheduler,
    num_epoch: int,
    num_gradient_update_batch: int,
    run_model_inference: Callable,
    loss_func: Callable,
    accelerator: Optional[Accelerator] = None,
    model_output_path: Optional[str] = None,
):
    # accelerator prep
    to_accelerate = accelerator is not None
    if to_accelerate:
        train_data_loader, eval_data_loader, model, optimizer = accelerator.prepare(
            train_data_loader,
            eval_data_loader,
            model,
            optimizer,
        )

    best_score = np.inf
    train_loss_history = []
    eval_loss_history = []

    progress_epoch = tqdm(range(num_epoch), total=num_epoch)
    for epoch_index in progress_epoch:
        progress_epoch.set_description(f"Epoch: {epoch_index}")
        epoch_train_loss = train_one_epoch(
            device,
            model,
            train_data_loader,
            optimizer,
            learning_rate_scheduler,
            num_gradient_update_batch,
            run_model_inference,
            loss_func,
            accelerator,
        )
        train_loss_history.append(epoch_train_loss)
        epoch_eval_loss = evaluate(
            device=device,
            model=model,
            tokenizer=tokenizer,
            data_loader=eval_data_loader,
            run_model_inference=run_model_inference,
            loss_func=loss_func,
        )

        eval_loss_history.append(epoch_eval_loss)

        if np.isfinite(epoch_eval_loss) and epoch_eval_loss < best_score:
            best_score = epoch_eval_loss
            if model_output_path is not None:
                save_pytorch_model(model, output_path=model_output_path)

        msg: str = f"Echo: {epoch_index} Loss {epoch_eval_loss}"
        logger.info(msg)


@torch.no_grad()
@inference_wrapper
def evaluate(
    device: torch.device,
    model: Module,
    tokenizer: GemmaTokenizerFast,
    data_loader: DataLoader,
    run_model_inference: Callable,
    loss_func: Callable,
) -> float:
    eval_loss: List[float] = []

    for batch in data_loader:
        inputs = get_model_inputs(device, batch)
        targets = get_model_targets(device, batch)
        batch_outputs, loss = get_pred_and_loss(device, model, inputs, run_model_inference, loss_func, targets)
        eval_loss.append(loss.nanmean().item())

    return np.nanmean(eval_loss)


def save_pytorch_model(model: Module, output_path: str):
    torch.save(model.state_dict(), output_path)


def get_pred_and_loss(
    device: torch.device,
    model: Module,
    inputs: Dict[str, torch.Tensor],
    run_model_inference: Callable,
    loss_func: Callable,
    target: Optional[torch.Tensor] = None,
    enable_autocast: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with autocast(device_type=device.type, enabled=enable_autocast):
        outputs = run_model_inference(model=model, **inputs)
        loss = loss_func(outputs, target) if target is not None else None
        return outputs, loss


def sharpened_consine_similarity_loss(
    inputs: torch.Tensor, targets: torch.Tensor, power: int = 3,
) -> torch.Tensor:
    cosine_similarity: torch.Tensor = torch.cosine_similarity(inputs.float(), targets.float())
    sign = torch.sign(cosine_similarity)
    csc: torch.Tensor = sign * torch.pow(torch.abs(cosine_similarity), power)
    loss: torch.Tensor = torch.nanmean(1 - csc)
    loss.requires_grad = True
    return loss
