import logging
from abc import ABC, abstractmethod

from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error, r2_score,\
                            accuracy_score, precision_score,\
                            recall_score, f1_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation
class ModelEvaluation(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Abstract method to evaluate a model.

        Args:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass


# Regression Model Evaluation
#------------------------------------------------------------------------------
# Class to define methods to evaluate regression model
class RegressionModelEvaluation(ModelEvaluation):
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Evaluates a regression model using R-squared and Mean Squared Error.

        Args:
        model (RegressorMixin): The trained regression model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing R-squared and Mean Squared Error.
        """
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {"evaluation_mse": mse, "evaluation_r2": r2}

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics
    
# Classification Model Regresson
#------------------------------------------------------------------------
# Class to define methods to evaluate classification model
class ClassificationModelEvaluation(ModelEvaluation):
    def evaluate_model(
        self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Evaluates a classification model using standard classification metrics.

        Args:
        model (ClassifierMixin): The trained classification model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The true labels for testing data.

        Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
        """
        logging.info("Predicting using the trained classification model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating classification metrics.")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        metrics = {
            "evaluation_Accuracy": accuracy,
            "evaluation_Precision": precision,
            "evaluation_Recall": recall,
            "evaluation_F1-Score": f1
        }

        logging.info(f"Classification Evaluation Metrics: {metrics}")
        return metrics


# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluation):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Args:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluation):
        """
        Sets a new strategy for the ModelEvaluator.

        Args:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: Union[RegressorMixin, ClassifierMixin], X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Executes the model evaluation using the current strategy.

        Args:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


