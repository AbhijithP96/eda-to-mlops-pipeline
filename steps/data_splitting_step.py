import pandas as pd
from src.data_splitter import DataSplitter,BasicDataSplit
from zenml import step

from typing import Tuple

@step
def data_splitting_step(df: pd.DataFrame, target_column: str , split_method: str = 'basic') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""

    if split_method == 'basic':
        splitter = DataSplitter(BasicDataSplit())

    else:
        raise ValueError(f'Unknown Splitter Method')
    
    X_train, X_test, y_train, y_test = splitter.split_data(df, target_column)
    
    return X_train, X_test, y_train, y_test