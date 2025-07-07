from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Class for Mutli Variate Analyis
# --------------------------------------------------------------
# This class define a template for multi variate analysis
class MultiVariateAnalysis(ABC):
    def analyse(self, df: pd.DataFrame) -> None:
        """Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.

        Args:
            df (pd.DataFrame): Dataframe to be analysed
        """

        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Generate and display a heatmap of the correlations between features.

        Args:
            df (pd.DataFrame): Dataframe to be analysed
        """
        pass
    
    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame) -> None:
        """Generate and display a pair plot of the selected features.

        Args:
            df (pd.DataFrame): Dataframe to be analysed
        """
        pass


# Multi Variate Analyser
#---------------------------------------------------------------------
# This class define the method to identify and visualize the missing values
class MultiVariateAnalyser(MultiVariateAnalysis):
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Generate and display a heatmap of the correlations between features."""

        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame) -> None:
        """Generate and display a pair plot of the selected features."""

        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()


# example usage
if __name__ == '__main__':

    df = pd.read_csv('./extracted/AmesHousing.csv')

    analyser = MultiVariateAnalyser()

    # Select important features for pair plot
    selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]
    analyser.analyse(selected_features)