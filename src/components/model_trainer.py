import os
import sys
import pickle
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model training started")

            # You said RandomForest performs better
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            logging.info(f"Model Accuracy: {accuracy}")
            logging.info(f"ROC-AUC Score: {roc_auc}")

            print("Model Accuracy:", accuracy)
            print("ROC-AUC Score:", roc_auc)

            # Save model
            os.makedirs("artifacts", exist_ok=True)

            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)

            logging.info("Model saved successfully")

            return roc_auc

        except Exception as e:
            raise CustomException(e, sys)