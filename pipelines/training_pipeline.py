from steps.data_loader_step import data_loader_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_handling_step import outlier_detection_step
from steps.data_splitting_step import data_splitting_step


from zenml import Model, pipeline, step

@pipeline(
    model=Model(
        name='price_predictor'),
)
def ml_pipeline(kwargs):

    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    #file_path = './data/archive.zip'
    raw_data = data_loader_step(kwargs['file_path'])

    # Handling Missing Values
    missing_value_strat = kwargs['missing_value_strat']
    handled_data = handle_missing_values_step(raw_data, missing_value_strat)

    # Feature Engineering
    feature_engineer_strat = kwargs['feature_engineer_strat']
    features = kwargs['engineer_features']
    transformed_data = feature_engineering_step(handled_data, strategy=feature_engineer_strat, features=features)

    # Outlier Detection
    detection_method = kwargs['outlier_detection']
    cleaned_data = outlier_detection_step(transformed_data, detection_method, kwargs)

    # Split Dataset
    target_column = kwargs['target_column']
    X_train, X_test, y_train, y_test = data_splitting_step(cleaned_data, target_column)

if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()