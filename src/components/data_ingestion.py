import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion method started")

            # ✅ Load dataset
            df = pd.read_csv("data/Churn_Modelling.csv")
            logging.info("Dataset loaded successfully")

            # ✅ Create artifacts folder
            os.makedirs("artifacts", exist_ok=True)

            # ✅ Save raw data (good practice)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # ✅ Stratified Train-Test Split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["Exited"]  # Important for churn dataset
            )

            # ✅ Save split data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train and test data saved successfully")
            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred in Data Ingestion")
            raise CustomException(e, sys)