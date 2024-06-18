import os
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_feature_name_group(feature_name: str) -> str:
    suffix = feature_name[feature_name.rfind("_") + len("_"):]
    if suffix.isnumeric():
        return feature_name[:feature_name.rfind("_")]
    else:
        return feature_name


def plot_feature_importance(
    feature_importance_data: pd.DataFrame,
    feature_name_col: str,
    feature_group_name: str,
) -> None:
    feature_names = {
        int(feature_name[feature_name.rfind("_") + len("_"):]): feature_name
        for feature_name in feature_importance_data[feature_name_col]
        if feature_name.find(feature_group_name) == 0 and feature_name[feature_name.rfind("_") + len("_"):].isnumeric()
    }
    sorted_feature_names = [
        feature_names[feature_name_suffix] for feature_name_suffix in sorted(feature_names)
    ]
    feature_importance_values = [
        feature_importance_data.loc[feature_importance_data["feature"] == feature_name, "median_fi"].values[0]
        for feature_name in sorted_feature_names
    ]
    plt.imshow(np.array(feature_importance_values).reshape((1, -1)))
    plt.savefig(f"/home/lkang/Downloads/lmsys-chatbot-arena/fi/{feature_group_name}.jpg")


def test_analyze_feature_importance() -> None:
    feature_names = set()
    parent_path = Path("/home/lkang/Downloads/")
    input_file_paths = [
        os.path.join(parent_path, "fi_0 (1).csv"),
        os.path.join(parent_path, "fi_1 (1).csv"),
        os.path.join(parent_path, "fi_2 (1).csv"),
        os.path.join(parent_path, "fi_3 (1).csv"),
    ]
    feature_group_importances = defaultdict(list)
    feature_importances = defaultdict(list)
    for file_path in input_file_paths:
        feature_importance_data = pd.read_csv(file_path)
        for i in range(len(feature_importance_data)):
            feature_name = feature_importance_data.iloc[i, :]["key"]
            feature_importance_value = feature_importance_data.iloc[i, :]["feature_importance"]
            feature_group = extract_feature_name_group(feature_name)
            feature_group_importances[feature_group].append(feature_importance_value)
            feature_importances[feature_name].append(feature_importance_value)

    feature_group_importance_data = []
    for feature_group in sorted(feature_group_importances):
        feature_importance_values = feature_group_importances[feature_group]
        feature_group_importance_data.append(
            [
                feature_group,
                np.min(feature_importance_values),
                np.median(feature_importance_values),
                np.mean(feature_importance_values),
                np.max(feature_importance_values),
            ]
        )
    df = pd.DataFrame(
        feature_group_importance_data,
        columns=["feature_group", "min_fi", "median_fi", "mean_fi", "max_fi"],
    )
    df.to_csv("/home/lkang/Downloads/feature_group_fi.csv", index=False)

    feature_importance_data = []
    for feature_group in sorted(feature_importances):
        feature_importance_values = feature_importances[feature_group]
        feature_importance_data.append(
            [
                feature_group,
                np.min(feature_importance_values),
                np.median(feature_importance_values),
                np.mean(feature_importance_values),
                np.max(feature_importance_values),
            ]
        )
    df = pd.DataFrame(
        feature_importance_data,
        columns=["feature", "min_fi", "median_fi", "mean_fi", "max_fi"],
    )
    df.to_csv("/home/lkang/Downloads/feature_fi.csv", index=False)

    for feature_group in feature_group_importances:
        plot_feature_importance(df, "feature", feature_group)
