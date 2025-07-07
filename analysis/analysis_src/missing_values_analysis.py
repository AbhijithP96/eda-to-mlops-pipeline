from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Class for Missing Values Analysis
# --------------------------------------------------------------
# This class define a template for missing values analysis
class MissingValuesAnalysis(ABC):
    def analyse(self, df: pd.DataFrame) -> None:
        """Performs a complete missing values analysis by identifying and visualizing missing values.

        Args:
            df (pd.DataFrame): Dataframe to be analysed
        """

        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        """Identifies the missing values in the dataframe

        Args:
            df (pd.DataFrame): Dataframe to be analysed
        """
        pass
    
    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        """Visualize the missing values in the dataframe

        Args:
            df (pd.DataFrame): Dataframe to be analysed
        """
        pass


# Missing Values Analyser
#---------------------------------------------------------------------
# This class define the method to identify and visualize the missing values
class MissingValueAnalyser(MissingValuesAnalysis):
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        """Prints the counts of missing values in each column"""

        print('\nMissing Values Count By Column')
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        """Visualize the missing values in the dataframe"""

        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()

# example usage
if __name__ == '__main__':

    df = pd.read_csv('./extracted/AmesHousing.csv')

    missing_analyser = MissingValueAnalyser()
    missing_analyser.analyse(df)


        