import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

from typing import Tuple

# Setup logging configuration
logging.basicConfig(level=logging.INFO, \
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Data Splitting
# -----------------------------------------------
# This class defines a common interface for different data splitting strategies.
class DataSplitting(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Abstract method to split the data into training and testing sets.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass


# Basic Train Test Split
#-------------------------------------------------------------
# Class to implement methods for a basic train test split of the dataset
class BasicDataSplit(DataSplitting):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initializes the BasicDataSplit with specific parameters.

        Args:
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): The seed used by the random number generator. Defaults to 42.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits the data into training and testing sets using a simple train-test split.

        Args:
            df (pd.DataFrame): The input DataFrame to be split.
            target_column (str):  The name of the target column.

        Returns:
            X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info(f"Basic Splitting data into train and test set with test data proportion of {self.test_size}")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        logging.info('Basic Data Splitting Completed')

        return X_train, X_test, y_train, y_test
    

# Data Splitter
#------------------------------------------------------------
# Class to switch between different data splitting strategies
class DataSplitter:
    def __init__(self, strategy: DataSplitting):
        """Initializes the DataSplitter with a specific data splitting strategy.

        Args:
            strategy (DataSplitting): The strategy to be used for data splitting.
        """
        self.strategy = strategy

    def set_new_strategy(self, strategy: DataSplitting) -> None:
        """Sets a new strategy for the DataSplitter.

        Args:
            strategy (DataSplitting): The new strategy to be used for data splitting.
        """
        logging.info("Switching data splitting strategy.")
        self.strategy = strategy

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """_summary_

        Args:
            df (pd.DataFrame): The input DataFrame to be split.
            target_column (str): The name of the target column.


        Returns:
            X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Splitting data using the selected strategy.")
        return self.strategy.split_data(df, target_column)
    

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('./extracted/AmesHousing.csv')

    data_splitter = DataSplitter(BasicDataSplit(test_size=0.2, random_state=42))
    X_train, X_test, y_train, y_test = data_splitter.split_data(df, target_column='SalePrice')
