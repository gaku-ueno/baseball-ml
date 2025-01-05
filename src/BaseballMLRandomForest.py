import numpy as np
import pandas as pd

import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import numpy as np

from BaseballMLPipeline import BaseballMLPipeline

class BaseballMLRandomForest(BaseballMLPipeline):
    def __init__(self, organzed_baseball_data: pd.DataFrame, model = RandomForestRegressor()) -> None:
        super().__init__(organzed_baseball_data, model)
        self.model = model

    def rand_forest_hyper_param_tuner(self):
        param_distributions = {
            'n_estimators': [n for n in range(100, 600, 100)],
            'max_features': ['sqrt', 'log2', 0.2, 0.3],
            'max_depth': [None]+ [n for n in range(10, 110, 10)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        random_search = RandomizedSearchCV(
            estimator= self.model,
            param_distributions=param_distributions,
            n_iter=20,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )

        random_search.fit(self.X_train, self.y_train)
        best_model = random_search.best_estimator_

        return best_model 