import pickle
from typing import Dict

import lightgbm as lgbm
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.utils import time_it


class LGBMClassifierPipeline:
    def __init__(self, config: Dict) -> None:
        self._config = config
        self.estimator = lgbm.LGBMClassifier(**self.config["lgbm"]["params"])
        self.num_scaler = RobustScaler()
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore")

    @property
    def config(self) -> Dict:
        return self._config

    def _preprocess_train(self, model_data: ModelData) -> np.ndarray:
        num_x = np.zeros((model_data.num_rows, 0))
        cat_x = np.zeros((model_data.num_rows, 0))
        if model_data.has_data_type(DataType.NUM):
            num_x = model_data.get_x_by_data_type(DataType.NUM)
            self.num_scaler.fit(num_x)
            num_x = self.num_scaler.transform(num_x)
        if model_data.has_data_type(DataType.CAT):
            cat_x = model_data.get_x_by_data_type(DataType.CAT)
            self.cat_encoder.fit(cat_x)
            cat_x = self.cat_encoder.transform(cat_x).toarray()

        return np.concatenate((num_x, cat_x), axis=1)

    def _preprocess_inference(self, model_data: ModelData) -> np.ndarray:
        num_x = np.zeros((model_data.num_rows, 0))
        cat_x = np.zeros((model_data.num_rows, 0))
        if model_data.has_data_type(DataType.NUM):
            num_x = model_data.get_x_by_data_type(DataType.NUM)
            num_x = self.num_scaler.transform(num_x)
        if model_data.has_data_type(DataType.CAT):
            cat_x = model_data.get_x_by_data_type(DataType.CAT)
            cat_x = self.cat_encoder.transform(cat_x).toarray()

        return np.concatenate((num_x, cat_x), axis=1)

    def save(self) -> None:
        with open(self.config['output_path'], "wb") as output_file:
            pickle.dump(self, output_file)

    def load(self) -> "LGBMClassifierPipeline":
        with open(self.config['output_path'], "rb") as output_file:
            return pickle.load(self, output_file)

    @time_it
    def fit(self, model_data: ModelData) -> "LGBMClassifierPipeline":
        x = self._preprocess_train(model_data)
        x = csr_matrix(x)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            model_data.y,
            test_size=self.config["lgbm"]["eval_pct"],
            random_state=self.config["seed"],
        )
        self.estimator.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test)],
            eval_metric="multi_logloss",
            callbacks=[
                lgbm.early_stopping(
                    stopping_rounds=self.config["lgbm"]["early_stopping"],
                    verbose=False,
                )
            ]
        )
        self.save()
        return self

    @time_it
    def predict_proba(self, model_data: ModelData) -> np.ndarray:
        x = self._preprocess_inference(model_data)
        return self.estimator.predict_proba(x)


class LGBMClassifierCVBlendingPipeline(LGBMClassifierPipeline):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.estimator = None
        self.estimators = []

    def get_feature_importance(self) -> Dict[str, np.array]:
        return {
            est_idx: estimator.feature_importances_
            for est_idx, estimator in enumerate(self.estimators)
        }

    @time_it
    def fit(self, model_data: ModelData) -> "LGBMClassifierPipeline":
        x = self._preprocess_train(model_data)
        x = csr_matrix(x)
        kfolds = StratifiedKFold(n_splits=4, random_state=self.config["seed"], shuffle=True)
        for i, (train_index, test_index) in enumerate(kfolds.split(x, model_data.y)):
            print(f">>>>>>>>>>>>>>>>>>>>CV: {i}")
            x_train = x[train_index]
            y_train = model_data.y[train_index]
            x_test = x[test_index]
            y_test = model_data.y[test_index]
            estimator = lgbm.LGBMClassifier(**self.config["lgbm"]["params"])
            estimator.fit(
                x_train,
                y_train,
                eval_set=[(x_test, y_test)],
                eval_metric="multi_logloss",
                callbacks=[
                    lgbm.early_stopping(
                        stopping_rounds=self.config["lgbm"]["early_stopping"],
                        verbose=False,
                    ),
                    lgbm.log_evaluation(period=100),
                ],
            )
            self.estimators.append(estimator)
        self.save()
        return self

    @time_it
    def predict_proba(self, model_data: ModelData) -> np.ndarray:
        x = self._preprocess_inference(model_data)

        pred_proba_outputs = []
        for estimator in self.estimators:
            pred_proba_outputs.append(estimator.predict_proba(x))

        return np.mean(pred_proba_outputs, axis=0)
