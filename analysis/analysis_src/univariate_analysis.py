from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Abstract Class for Univariate Analysis
# --------------------------------------------------------------
# This class define a template for univariate analysis
class UnivariateAnalysis(ABC):
    @abstractmethod
    def analyse(self, df: pd.DataFrame, feature: str) -> None:
        """Perform univariate analysis on a specific feature of the dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing the data
            feature (str): The name of the feature to be analysed.
        """

        pass

# Numerical Feature Analysis
#------------------------------------------------------------------------------------------
# This class define the method to analyse numerical features by plotting their distribiution
class NumericalUnivariateAnalysis(UnivariateAnalysis):
    def analyse(self, df: pd.DataFrame, feature: str) -> None:
        """
        Plots the distribution of a numerical feature using a histogram and KDE.

        """

        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], kde=True, bins=30) #kde=True adds a smooth curve that estimates the distribution of the data.
        plt.title(f'Distrbution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()


# Categorical Feature Analysis
#------------------------------------------------------------------------------------------
# This class define the method to analyse categorical features by plotting their frequence/count distribiution
class CategoricalUnivariateAnalysis(UnivariateAnalysis):
    def analyse(self, df: pd.DataFrame, feature: str) -> None:
        """
        Plots the distribution of a categorical feature using a bar plot.
        """

        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette='muted')
        plt.title(f'Distrbution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()

# UniVariate Analyser
#------------------------------------------------------------------------------------------
# This class allows the switching between different univariate analysis.
class UnivariateAnalyser:
    def __init__(self, analysis: UnivariateAnalysis):
        """Initialize the analyzer with a specific univariate analysis

        Args:
            analysis (UnivariateAnalysis): The univariate analysis to be used.
        """
        
        self.analysis = analysis

    def set_analysis(self, analysis: UnivariateAnalysis) -> None:
        """Set a new univariate analysis.

        Args:
            analysis (UnivariateAnalysis): The new univariate analysis to be used
        """

        self.analysis = analysis

    def execute_analysis(self, df: pd.DataFrame, feature: str) -> None:
        """Executes the specific univariate analysis

        Args:
            df (pd.DataFrame): Dataframe containing the data
            feature (str): The name of the feature to be analysed.
        """

        self.analysis.analyse(df, feature)


# example usage
if __name__ == '__main__':

    df = pd.read_csv('./extracted/AmesHousing.csv')

    missing_analyser = UnivariateAnalyser(NumericalUnivariateAnalysis())
    missing_analyser.execute_analysis(df, 'SalePrice')

    missing_analyser.set_analysis(CategoricalUnivariateAnalysis())
    missing_analyser.execute_analysis(df, 'Neighborhood')