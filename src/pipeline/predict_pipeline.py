import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            # load saved object (model.pkl expected to be dict {"preprocessor":..., "model":...})
            model_obj = load_object(file_path=model_path)

            if isinstance(model_obj, dict):
                model = model_obj.get('model', None)
                preprocessor = model_obj.get('preprocessor', None)
            else:
                # backward compatibility: if a plain model was saved
                model = model_obj
                preprocessor = None

            # if preprocessor wasn't saved inside model.pkl, try loading preprocessor.pkl
            if preprocessor is None:
                try:
                    preprocessor = load_object(file_path=preprocessor_path)
                except Exception:
                    preprocessor = None

            if model is None:
                raise CustomException("Model not found in artifacts/model.pkl", sys)

            # ensure features is a pandas DataFrame
            if isinstance(features, (dict, list, tuple, np.ndarray)):
                features = pd.DataFrame(features)
            if not isinstance(features, pd.DataFrame):
                raise CustomException("features must be a pandas DataFrame or convertible to one", sys)

            # Apply preprocessing if available
            if preprocessor is not None:
                data_transformed = preprocessor.transform(features)
            else:
                # if no preprocessor, try to pass raw features (may fail)
                data_transformed = features.values

            preds = model.predict(data_transformed)

            # return plain python list for safe JSON/Jinja use
            return np.array(preds).tolist()

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)