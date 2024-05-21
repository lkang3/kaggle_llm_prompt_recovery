import pytest
import torch
import numpy as np
from torch.nn import Parameter
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification

from kaggle_lmsys.models import get_diff_of_two_response


def test_get_diff_of_two_response():
    batch_size = 10
    sentence_size = 7
    embedding_size = 128
    x = torch.rand((batch_size, sentence_size, embedding_size))
    mask_values = np.random.choice([0, 1, 2], sentence_size)
    mask = torch.tensor([mask_values] * batch_size)
    get_diff_of_two_response(x, mask)


def test_last_hidden_state():
    model_name = "microsoft/deberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    texts = ["This is a prompt", "This is text from model a", "This is text from model b"]
    text_tokens = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**text_tokens)
        lhs = outputs.last_hidden_state
        assert lhs.shape == (3, 7, 768)
