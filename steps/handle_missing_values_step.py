import pandas as pd
from src.handle_missing_values import MissingValueHandler, DropMissingValues, FillMissingValues
from zenml import step

@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """Handle Missing Values in the given dataframe based on the strategy"""

    if strategy == 'drop':
        handler = MissingValueHandler(DropMissingValues())
    elif strategy in ['mean', 'median', 'mode', 'constant']:
        handler = MissingValueHandler(FillMissingValues())
    else:
        raise ValueError(f'\n Unknown strategy for handling missing values; {strategy}')
    
    df_handled = handler.handle_missing_values(df)
    return df_handled

