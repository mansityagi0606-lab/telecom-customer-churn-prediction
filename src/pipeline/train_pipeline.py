from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
import os

class TrainPipeline:

    def run_pipeline(self):

        logging.info("Pipeline execution started")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
            train_path, test_path
        )

        trainer = ModelTrainer()
        acc = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

        print("Model Accuracy:", acc)

        logging.info("Pipeline execution finished")


if __name__ == "__main__":
    print("Training pipeline started...")
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
    print("Training pipeline finished.")