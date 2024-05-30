import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from kaggle_lmsys.utils import tokenization_prompt_one_resp
from kaggle_lmsys.utils import get_device
from kaggle_lmsys.utils import Collator
from kaggle_lmsys.models.deberta_classifier import CustomizedDetertaClassifier

SEED = 123
random_state = np.random.RandomState(SEED)
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)
numpy_gen = np.random.Generator(np.random.PCG64(seed=SEED))
device = get_device()


def predict_model_types(
    tokenizer_path: str,
    model_path: str,
    data: pd.DataFrame,
    prompt_field_name: str,
    resp_field_name: str,
    tokenization_max_length: int,
) -> np.array:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = CustomizedDetertaClassifier.from_pretrained(model_path)
    model.to(device)

    dataset = Dataset.from_pandas(data)
    dataset.cleanup_cache_files()
    tokenization_args = {
        "tokenizer": tokenizer,
        "max_length": tokenization_max_length,
        "prompt_field": prompt_field_name,
        "resp_field": resp_field_name,
        "target_field": None,
    }
    dataset = dataset.map(
        function=tokenization_prompt_one_resp,
        batched=False,
        fn_kwargs=tokenization_args,
        remove_columns=dataset.column_names,
    )
    data_collator = Collator(
        tokenizer,
        max_length=tokenization_max_length,
    )

    model_types = []
    for row_id, data in enumerate(dataset):
        preds = model(**data_collator([data]))
        preds = torch.nn.functional.softmax(preds.logits.cuda(), dim=-1)
        model_type = torch.argmax(preds, axis=1)
        model_types.append(model_type)

    model_types = torch.vstack(model_types).flatten()
    return model_types.detach().cpu().numpy()
