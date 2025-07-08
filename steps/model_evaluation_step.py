import logging
from typing import Annotated

import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.model_evaluator import ModelEvaluator, RegressionModelEvaluation, ClassificationModelEvaluation

from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

# setup experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def model_evaluation_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
):
    """Evaluates the trained model using ModelEvaluator"""

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Applying the same preprocessing to the test data.")

    # Apply the preprocessing and model prediction
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    # build the evaluator model depending on the model
    if isinstance(trained_model.named_steps['model'], LinearRegression):
        evaluator = ModelEvaluator(RegressionModelEvaluation())
    elif isinstance(trained_model.named_steps['model'], LogisticRegression):
        evaluator = ModelEvaluator(ClassificationModelEvaluation())
    else:
        raise ValueError('Unkown Trained Model Type')

    # train the model
    evaluation_metric = evaluator.evaluate(trained_model.named_steps['model'],
                                            X_test_processed, y_test)


    logging.info(f"Evaluation metrics: {evaluation_metric}")

    return evaluation_metric
    
