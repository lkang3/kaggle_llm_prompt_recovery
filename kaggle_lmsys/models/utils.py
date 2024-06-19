from typing import List

import numpy as np

from kaggle_lmsys.models.entities import ModelData


def merge_model_data(model_data_list: List[ModelData]) -> ModelData:
    merged_x = np.concatenate([model_data.x for model_data in model_data_list], axis=1)
    merged_data_types = []
    for model_data in model_data_list:
        merged_data_types.extend(model_data.data_types)
    merged_col_names = []
    for model_data in model_data_list:
        merged_col_names.extend(model_data.col_names)

    return ModelData(
        x=merged_x,
        data_types=merged_data_types,
        col_names=merged_col_names,
    )
