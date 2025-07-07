import pandas as pd
from src.feature_engineering import FeatureEngineer, LogTransformation, \
                                    StandardScalerTransformation, MinMaxScalerTransformation, OneHotEncoding
from zenml import step

@step
def feature_engineering_step(df: pd.DataFrame, strategy: str = 'log', features: list = None) -> pd.DataFrame:
    """Apply Feature Engineering to the dataframe using the given strategy"""

    if features is None:
        raise ValueError('No features is specified')
    
    if strategy == 'log':
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == 'std':
        engineer = FeatureEngineer(StandardScalerTransformation(features))
    elif strategy == 'minmax':
        engineer = FeatureEngineer(MinMaxScalerTransformation(features))
    elif strategy == 'onehot':
        engineer = FeatureEngineer(OneHotEncoding(features))
    else:
        raise ValueError(f'Unknown Feature Engineering Strategy :{strategy}')
    
    return engineer.apply_feature_engineering(df)