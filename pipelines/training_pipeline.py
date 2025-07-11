from steps.data_loader_step import data_loader_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_handling_step import outlier_detection_step
from steps.data_splitting_step import data_splitting_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step

from zenml import Model, pipeline, step

@pipeline(
    model=Model(
        name='price_predictor'),
)
def ml_pipeline(kwargs):

    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    #file_path = './data/archive.zip'
    raw_data = data_loader_step(kwargs['data']['file_path'])

    # Handling Missing Values
    pipeline_params = kwargs['parameters']
    missing_value_strat = pipeline_params['missing_value_strat']
    handled_data = handle_missing_values_step(raw_data, missing_value_strat)

    # Feature Engineering
    feature_engineer_strat = pipeline_params['feature_engineer_strat']
    features = pipeline_params['engineer_features']
    transformed_data = feature_engineering_step(handled_data, strategy=feature_engineer_strat, features=features)

    # Outlier Detection
    column_name = pipeline_params['column_name']
    cleaned_data = outlier_detection_step(transformed_data, column_name , pipeline_params)

    # Split Dataset
    target_column = pipeline_params['target_column']
    X_train, X_test, y_train, y_test = data_splitting_step(cleaned_data, target_column)

    # Build and train model
    model_name = kwargs['model']['name']
    model = model_building_step(X_train, y_train, model_name)

    # evaluate model
    evaluation_metric = model_evaluation_step(model, X_test, y_test)

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()