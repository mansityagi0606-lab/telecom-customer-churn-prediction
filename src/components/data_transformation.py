import sys
import os
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_transformer(self, df):

        # Drop same columns as notebook
        df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

        X = df.drop("Exited", axis=1)

        categorical_columns = X.select_dtypes(include="object").columns.tolist()
        numerical_columns = X.select_dtypes(exclude="object").columns.tolist()

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        preprocessor = self.get_transformer(train_df)

        train_df = train_df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
        test_df = test_df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

        X_train = train_df.drop("Exited", axis=1)
        y_train = train_df["Exited"]

        X_test = test_df.drop("Exited", axis=1)
        y_test = test_df["Exited"]

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        os.makedirs("artifacts", exist_ok=True)

        with open(self.transformation_config.preprocessor_obj_file_path, "wb") as f:
            pickle.dump(preprocessor, f)

        return (
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test
        )