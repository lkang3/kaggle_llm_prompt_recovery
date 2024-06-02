from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd

from kaggle_lmsys.models.enum import DataType


@dataclass
class ModelData:
    data_types: np.array
    x: np.ndarray
    y: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        if self.y is not None and len(self.x) != len(self.y):
                raise ValueError(f"x (length: {len(self.x)}) and y ({len(self.y)}) length mismatch")

    @property
    def num_rows(self) -> int:
        return self.x.shape[0]

    def has_data_type(self, data_type: DataType) -> bool:
        return data_type in self.data_types

    def get_x_by_data_type(self, data_type: DataType) -> np.ndarray:
        col_selector = self.data_types == data_type
        return self.x[:, col_selector]
