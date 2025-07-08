import logging
from abc import ABC, abstractmethod
from typing import Any, Union

import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import json

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Template
class ModelBuilding(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Union[RegressorMixin, ClassifierMixin]:
        """
        Abstract method to build and train a model.

        Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        pass


# Linear Regression Model
#----------------------------------------------------------------------------------
# Class to define methods to build and train linear regression model
class LinearRegressionModel(ModelBuilding):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains a linear regression model using scikit-learn.

        Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Linear Regression model with scaling.")

         # Identify categorical and numerical columns
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
        numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

        logging.info(f"Categorical columns: {categorical_cols.tolist()}")
        logging.info(f"Numerical columns: {numerical_cols.tolist()}")

        # Define preprocessing for categorical and numerical features
        numerical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Define the model training pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train, y_train)  # Fit the pipeline to the training data

        logging.info("Model training completed.")

        # Log the columns that the model expects
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )
        # Write to JSON
        with open("expected_columns.json", "w") as f:
            json.dump({"expected_columns": expected_columns}, f, indent=4)

        logging.info(f"Model expects the following columns: {expected_columns}")

        return pipeline
    

# Logistic Regression Model
#-------------------------------------------------------------------------------------
# Class to define methods to build and train logistic regression model
class LogisticRegressionModel(ModelBuilding):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains a logistic regression model using scikit-learn.

        Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Logistic Regression model.")

         # Identify categorical and numerical columns
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
        numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

        logging.info(f"Categorical columns: {categorical_cols.tolist()}")
        logging.info(f"Numerical columns: {numerical_cols.tolist()}")

        # Define preprocessing for categorical and numerical features
        numerical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Define the model training pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LogisticRegression())])

        logging.info("Training Logistic Regression model.")
        pipeline.fit(X_train, y_train)  # Fit the pipeline to the training data

        logging.info("Model training completed.")

        # Log the columns that the model expects
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )

        # Write to JSON
        with open("expected_columns.json", "w") as f:
            json.dump({"expected_columns": expected_columns}, f, indent=4)

        logging.info(f"Model expects the following columns: {expected_columns}")

        return pipeline
    

# Context Class for Model Building
class ModelBuilder:
    def __init__(self, strategy: ModelBuilding):
        """
        Initializes the ModelBuilder with a specific model building strategy.

        Args:
        strategy (ModelBuildingStrategy): The strategy to be used for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuilding):
        """
        Sets a new strategy for the ModelBuilder.

        Args:
        strategy (ModelBuildingStrategy): The new strategy to be used for model building.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Union[RegressorMixin, ClassifierMixin]:
        """
        Executes the model building and training using the current strategy.

        Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)


# Example usage
if __name__ == "__main__":
    
    df = pd.read_csv('./extracted/AmesHousing.csv')
    X_train = df.drop(columns=['SalePrice'])
    y_train = df['SalePrice']

    model_builder = ModelBuilder(LinearRegressionModel())
    trained_model = model_builder.build_model(X_train, y_train)
    print(trained_model.named_steps['model'].coef_)  # Print model coefficients
