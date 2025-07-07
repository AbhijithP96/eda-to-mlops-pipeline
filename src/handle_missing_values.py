import logging
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract class to define the template for handling missing values
class HandleMissingValues(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method to handle missing values in the dataframe.

        Args:
            df (pd.DataFrame): The dataframe containing missing values.

        Returns:
            pd.DataFrame: The dataframe with missing values handled
        """
        pass


# Drop Missing Values
#----------------------------------------------------------------
# Class to define methods to drop missing values in the given dataframe
class DropMissingValues(HandleMissingValues):
    def __init__(self, axis=0, thresh=None):
        """Initializes DropMissingValues with specific parameters.

        Args:
            axis (int, optional): 0 to drop the rows with missing values, 1 to drop columns with missing values. Defaults to 0.
            thresh (_type_, optional): The threshold for non-NA values. Rows or columns with values less than thresh are dropped. Defaults to None.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method to drop  rows or cloumns with  missing values in the dataframe, based on axis and threshold.

        Args:
            df (pd.DataFrame): The dataframe containing missing values.

        Returns:
            pd.DataFrame: The dataframe with missing values dropped.
        """

        logging.info(f'Dropping Missing Values along the axis {self.axis} with threshold {self.thresh}')
        df_handled = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info('Missing Values Dropped')

        return df_handled
    

# Fill Missing Values
#-------------------------------------------------------------------------
# Class to define methods to fill missing values in the given dataframe
class FillMissingValues(HandleMissingValues):
    def __init__(self, method: str='mean', fill_value=None):
        """Initializes FillMissingValues with a specific method or fill value

        Args:
            method (str, optional): The method to fill missing values either : {mean, median, mode, constant}. Defaults to 'mean'.
            fill_value (any, optional): The constant value to be fill in for missing values when the method is constant. Defaults to None.
        """

        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills the missing values in the dataframe using the specified method

        Args:
            df (pd.DataFrame): The dataframe containing missing values.

        Returns:
            pd.DataFrame: The dataframe with missing values filled.
        """
        
        logging.info(f'Filling Missing Values using the method: {self.method}')

        df_handled = df.copy()
        if self.method == 'mean':
            numeric_columns = df_handled.select_dtypes(include='number').columns
            df_handled[numeric_columns] = df_handled[numeric_columns].fillna(value=df[numeric_columns].mean())

        elif self.method == 'median':
            numeric_columns = df_handled.select_dtypes(include='number').columns
            df_handled[numeric_columns] = df_handled[numeric_columns].fillna(value=df[numeric_columns].median())

        elif self.method == 'mode':
            for column in df_handled.columns:
                df_handled[column].fillna(value=df[column].mode().iloc[0], inplace=True)

        elif self.method == 'constant':
            df_handled = df_handled.fillna(value=self.fill_value)

        else:
            logging.warning(f'Unknown Method: {self.method}. No missing values filled')

        logging.info('Missing Values Filled')
        return df_handled
    

# Class for handling missing values
class MissingValueHandler:
    def __init__(self, handler_type: HandleMissingValues):
        """Initializes MissingValueHandler with a specific missing values strategy.

        Args:
            handler_type (HandleMissingValues): The missing value handling strategy.
        """
        self.handler_type = handler_type

    def set_type(self, handler_type: HandleMissingValues) -> None:
        """Sets a new strategy for handling missing values

        Args:
            handler_type (HandleMissingValues): The new strategy for handling missing values
        """
        self.handler_type = handler_type

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the missing value handling using the current strategy

        Args:
            df (pd.DataFrame): The dataframe containing missing values.

        Returns:
            pd.DataFrame: The dataframe with missing values handled.
        """

        logging.info('Executing Missing Value Handler')
        return self.handler_type.handle(df)
    
# example usage
if __name__ == '__main__':
    df = pd.read_csv('./extracted/AmesHousing.csv')

    handler = MissingValueHandler(DropMissingValues())
    df_cleaned = handler.handle_missing_values(df)

    handler.set_type(FillMissingValues())
    df_cleaned = handler.handle_missing_values(df)