import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from src.model_builder import ModelBuilder, LinearRegressionModel, LogisticRegressionModel 

from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name='price_predictor',
    description='Price predicton model for houses'
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series, model_name: str
    ) -> Annotated[Pipeline, ArtifactConfig(name='sklearn_pipeline', is_model_artifact=True)]:
    """Build and trains the model with the given data"""

    if model_name == 'Linear Regression':
        model = ModelBuilder(LinearRegressionModel())

    elif model_name == 'Logistic Regression':
        model = ModelBuilder(LogisticRegressionModel())

    else:
        raise ValueError(f'Unknown model name: {model}')


    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable autologging for sckikit-learn
        mlflow.sklearn.autolog()

        # train the model
        pipeline = model.build_model(X_train, y_train)


    except Exception as e:
        raise e
    
    finally:
        mlflow.end_run()

    return pipeline