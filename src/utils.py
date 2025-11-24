# ...existing code...
import os
import sys
import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e: 
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)   
    
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    """
    Trains/evaluates each model. If param contains a grid for a model name, GridSearchCV is applied.
    param is expected to be a dict mapping the exact model name (same as in `models`) to param grid.
    """
    try:
        report = {}
        for model_name, model in models.items():
            try:
                param_grid = param.get(model_name, {}) if isinstance(param, dict) else {}
            except Exception:
                param_grid = {}

            if param_grid:
                gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
                gs.fit(X_train, y_train)
                best_params = gs.best_params_
                model.set_params(**best_params)

            # Fit (either tuned or default)
            model.fit(X_train, y_train)

            # Predict testing data
            y_test_pred = model.predict(X_test)

            # Use R2 score for regression tasks
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
