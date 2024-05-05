import logging
import sys
import time

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def time_it(func: Callable) -> None:
    def _time_it(*args, **kwargs):
        start_perf_conter = time.perf_counter()
        outputs = func(*args, **kwargs)
        end_perf_conter = time.perf_counter()
        msg: str = f">>>>>>>>>>>>>> time it - Func:{func} Time:{end_perf_conter - start_perf_conter}"
        logger.info(msg)
        return outputs

    return _time_it


def inference_wrapper(func: Callable) -> None:

    def _wrapper(*args, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        logger.info(">>>>>>>>>>>>>>>> model.eval()")
        model.eval()
        outputs = func(*args, **kwargs)
        model.train()
        logger.info(">>>>>>>>>>>>>>>> model.train()")
        return outputs

    return _wrapper


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model_inputs(
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    is_inference: Optional[bool] = False,
    input_field_names: Optional[List[str]] = ["input_ids", "token_type_ids", "attention_mask"],  # FIXME
) -> Dict[str, torch.Tensor]:
    inputs: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in batch.items() if k in input_field_names}
    if not is_inference and "labels" in batch:  # FIXME
        inputs["labels"] = get_model_targets(device, batch)
    return inputs


def get_model_targets(
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    target_name: Optional[str] = "labels",  # FIXME
) -> Optional[torch.Tensor]:
    if target_name in batch:
        return batch[target_name].to(device)
    return None
