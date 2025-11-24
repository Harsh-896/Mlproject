# ...existing code...
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # feature lists — keep consistent with your CSV column names
        self.numeric_features = ['writing score', 'reading score']
        self.categorical_features = [
            'gender',
            'race/ethnicity',
            'parental level of education',
            'lunch',
            'test preparation course'
        ]

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformer object")

            # numeric pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # OneHotEncoder compatibility across sklearn versions
            try:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

            # categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot', ohe)
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, self.numeric_features),
                    ('cat_pipeline', cat_pipeline, self.categorical_features)
                ],
                remainder='drop'
            )

            logging.info("Preprocessor created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Starting data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math score'
            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise CustomException(f"Target column '{target_column_name}' missing in train/test files", sys)

            input_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_train_df = train_df[target_column_name]

            input_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_test_df = test_df[target_column_name]

            # verify required input columns are present
            missing_train = set(self.numeric_features + self.categorical_features) - set(input_train_df.columns)
            missing_test = set(self.numeric_features + self.categorical_features) - set(input_test_df.columns)
            if missing_train:
                raise CustomException(f"Missing columns in train data: {missing_train}", sys)
            if missing_test:
                raise CustomException(f"Missing columns in test data: {missing_test}", sys)

            input_train_arr = preprocessor_obj.fit_transform(input_train_df)
            input_test_arr = preprocessor_obj.transform(input_test_df)

            train_arr = np.c_[input_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            # save preprocessor
            save_path = self.data_transformation_config.preprocessor_obj_file_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_object(file_path=save_path, obj=preprocessor_obj)

            logging.info("Data transformation completed")
            return train_arr, test_arr, save_path

        except Exception as e:
            raise CustomException(e, sys)
# ...existing code...