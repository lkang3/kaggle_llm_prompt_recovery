import pytest
import torch
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from kaggle_lmsys.models.deberta_classifier import get_diff_of_two_response


@pytest.fixture
def data() -> pd.DataFrame:
    data = pd.read_csv("/home/lkang/Downloads/lmsys-chatbot-arena/train.csv")
    return data


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


def test_correlation_between_target_and_model_type_association(data: pd.DataFrame) -> None:
    p_value = 0.95
    combined_model_types = data.apply(lambda row: f"{row['model_a']}-vs-{row['model_b']}", axis=1)
    winner_model_a = data["winner_model_a"].values
    winner_model_b = data["winner_model_b"].values
    winner_tie = data["winner_tie"].values
    contingency_table = pd.crosstab(
        combined_model_types, [winner_model_a, winner_model_b, winner_tie]
    )
    stat, p, dof, expected = chi2_contingency(contingency_table)
    critical = chi2.ppf(p_value, dof)
    print(f"stat: {stat}, critical: {critical}")
    if abs(stat) >= critical:
        print("Dependent (reject H0)")
    else:
        print("Independent (fail to reject H0)")
    alpha = 1.0 - p_value
    print(f"p_value: {p}, alpha: {alpha}")
    print("significance=%.3f, p=%.3f" % (alpha, p))
    if p <= alpha:
        print("Dependent (reject H0)")
    else:
        print("Independent (fail to reject H0)")
