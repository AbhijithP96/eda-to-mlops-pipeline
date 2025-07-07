from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Abstract Class for Bivariate Analysis
# --------------------------------------------------------------
# This class define a template for Bivariate analysis
class BivariateAnalysis(ABC):
    @abstractmethod
    def analyse(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """Perform bivariate analysis on the two features of the dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing the data
            feature1 (str): The name of the first feature to be analysed.
            feature2 (str): The name of the second feature to be analysed.
        """

        pass

# Numerical vs Numerical Feature Analysis
#------------------------------------------------------------------------------------------
# This class define the method to analyse one numerical features against another numerrical feature by plotting their distribiution
class NumericalvsNumeriacalAnalysis(BivariateAnalysis):
    def analyse(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Plots the distribution of the two numerical features using a scatter plot

        """

        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f'Distrbution of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Categorical vs Numerical Feature Analysis
#------------------------------------------------------------------------------------------
# This class define the method to analyse categorical features against numerical feature by plotting their distribution
class CategoricalvsNumericalAnalysis(BivariateAnalysis):
    def analyse(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Plots the distribution of categorical and numerical feature using box plot.
        """

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f'Distrbution of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

# BiVariate Analyser
#------------------------------------------------------------------------------------------
# This class allows the switching between different bivariate analysis.
class BivariateAnalyser:
    def __init__(self, analysis: BivariateAnalysis):
        """Initialize the analyzer with a specific bivariate analysis

        Args:
            analysis (UnivariateAnalysis): The bivariate analysis to be used.
        """
        
        self.analysis = analysis

    def set_analysis(self, analysis: BivariateAnalysis) -> None:
        """Set a new bivariate analysis.

        Args:
            analysis (UnivariateAnalysis): The new bivariate analysis to be used
        """

        self.analysis = analysis

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """Executes the specific bivariate analysis

        Args:
            df (pd.DataFrame): Dataframe containing the data
            feature1 (str): The name of the first feature to be analysed.
            feature2 (str): The name of the second feature to be analysed.
        """

        self.analysis.analyse(df, feature1, feature2)


# example usage
if __name__ == '__main__':

    df = pd.read_csv('./extracted/AmesHousing.csv')

    missing_analyser = BivariateAnalyser(NumericalvsNumeriacalAnalysis())
    missing_analyser.execute_analysis(df, 'Gr Liv Area', 'SalePrice')

    missing_analyser.set_analysis(CategoricalvsNumericalAnalysis())
    missing_analyser.execute_analysis(df, 'Overall Cond', 'SalePrice')