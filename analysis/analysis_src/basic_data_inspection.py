from abc import ABC, abstractmethod

import pandas as pd

# Abstract Base Class for Data Inspection
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
class DataInspection(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """Perform a specific type of data inspection.

        Args:
            df (pd.DataFrame): Dataframe on which inspection is to be carried out.
        """
        pass

# Data Types Inspection
# --------------------------------------------
# This class inspects the data types of each column and counts non-null values.
class DataTypesInspection(DataInspection):
    def inspect(self, df: pd.DataFrame) -> None:
        """Inspects and prints the data types and non-null counts of the dataframe columns.

        Args:
            df (pd.DataFrame): Dataframe to be inspected.
        """
        print('\nData Types and Non-null Counts')
        print(df.info())


# Summary Statistics Inspection
# -----------------------------------------------------
# This class provides summary statistics for both numerical and categorical features.
class SummaryStatsInspection(DataInspection):
    def inspect(self, df: pd.DataFrame) -> None:
        """Prints summary statistics for numerical and categorical features.

        Args:
            df (pd.DataFrame): Dataframe to be inspected.
        """

        print('\nSummary Statistics (Numerical)')
        print(df.describe())
        print('\nSummary Statistics (Categorical)')
        print(df.describe(include=['O']))
        

# Data Inspector
# -----------------------------------------------------
# This class allows you to switch between different data inspection
class DataInspector:
    def __init__(self, inspection: DataInspection):
        """Initializes the DataInspector with a spection inspection.

        Args:
            inspection (DataInspection): The type of data inspection.
        """
        self.inspection = inspection

    def set_inspection(self, inspection: DataInspection) -> None:
        """Sets a new data inspection type.

        Args:
            inspection (DataInspection): The new data inspection type.
        """
        self.inspection = inspection

    def execute_inspection(self, df: pd.DataFrame) -> None:
        """Executes the data inspection using the current inspection type.

        Args:
            df (pd.DataFrame): Dataframe to be inspected.
        """
        self.inspection.inspect(df)


# example usage
if __name__ == '__main__':

    df = pd.read_csv('./extracted/AmesHousing.csv')

    inspector = DataInspector(DataTypesInspection())
    inspector.execute_inspection(df)

    inspector.set_inspection(SummaryStatsInspection())
    inspector.execute_inspection(df)