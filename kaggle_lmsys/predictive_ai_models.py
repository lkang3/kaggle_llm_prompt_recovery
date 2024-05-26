from typing import Dict

import lightgbm as lgbm
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


class LGBMBasedClassifier:
    def __init__(
        self,
        estimator_params: Dict,
        eval_pct: float = 0.2,
        seed: int = 123,
    ):
        self.estimator = lgbm.LGBMClassifier(**estimator_params)
        self.scaler = RobustScaler()
        self.estimator_config = estimator_params
        self.eval_pct = eval_pct
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.array) -> None:
        self.scaler.fit(x)
        x = self.scaler.transform(x)

        x = csr_matrix(x)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.eval_pct,
            random_state=self.seed,
        )

        self.estimator.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test)],
            eval_metric="multi_logloss",
            callbacks=[
                lgbm.early_stopping(stopping_rounds=50)
            ]
        )

    def predict_proba(self, x) -> np.ndarray:
        x = self.scaler.transform(x)
        return self.estimator.predict_proba(x)
