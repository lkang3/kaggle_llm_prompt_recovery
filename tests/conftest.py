import pytest
from pathlib import Path
import pandas as pd

from kaggle_lmsys.train_utils import add_target
from kaggle_lmsys.utils import clean_data


@pytest.fixture
def input_data_path() -> Path:
    return Path("/home/lkang/Downloads/lmsys-chatbot-arena/train.csv")


@pytest.fixture
def input_dataframe(input_data_path: Path) -> pd.DataFrame:
    input_fields = ["prompt", "response_a", "response_b"]
    df = clean_data(input_data_path, input_fields)
    df = add_target(
        df,
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
        "labels",
    )
    return df.loc[: 20, :]
