import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            print("Data Ingestion method starts")
            print("Trying to read dataset from notebook/data/stud.csv...")
            
            # Read the CSV file
            df = pd.read_csv('notebook/data/stud.csv')
            print('Dataset read as pandas DataFrame')
            print(f"Dataset shape: {df.shape}")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            print("Artifacts directory created")

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            print(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            # Train-test split
            print("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            print(f"Train data saved to {self.ingestion_config.train_data_path}")
            print(f"Test data saved to {self.ingestion_config.test_data_path}")
            print("Data ingestion completed successfully!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error during data ingestion: {e}")
            sys.exit(1)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"\nData ingestion completed!")
    print(f"Train path: {train_data}")
    print(f"Test path: {test_data}")
    # ...existing code that reads train_data, test_data paths...
    data_transformation = DataTransformation()
    # unpack three values (train_arr, test_arr, preprocessor_path)
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    r2 = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
    print(f"Model R2 score: {r2}")
    

