from analysis.analysis_src import DataInspector, DataTypesInspection, SummaryStatsInspection, MissingValueAnalyser,\
                            UnivariateAnalyser, NumericalUnivariateAnalysis, CategoricalUnivariateAnalysis, \
                            BivariateAnalyser, NumericalvsNumeriacalAnalysis, CategoricalvsNumericalAnalysis, \
                            MultiVariateAnalyser

import pandas as pd

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Class to select different data analysis on the given dataframe based on user interaction
class EDA:
    def __init__(self, df: pd.DataFrame):
        """Initializes with the dataframe on which analysis is to be carried out.

        Args:
            df (pd.DataFrame): DataFrame to be analysed
        """

        self.df = df

    def start_analysis(self) -> None:
        """Provides user with option for different data analysis.
        The user can stop the analysis at any time by selecting the appropriate option.

        Raises:
            ValueError: If option is not a valid selection.
        """

        analysis = [self.basic_data_inspection, self.missing_values_analysis,
                    self.univariate_analysis, self.bivariate_analysis,
                    self.multivariate_analysis]

        while True:

            print('\nSelect Analysis\n')
            print('\n1. Data Inspection\n' \
                        '2. Missing Values Analysis\n' \
                        '3. Univariate Analysis\n' \
                        '4. Bivariate Analysis\n' \
                        '5. Multivariate Analysis\n' \
                        '6. Exit')
            
            try:
                option = int(input())
                if option < 0 or option > 6:
                    raise ValueError
            except ValueError:
                print('\nInvalid Selction, Please re-enter')

            if option == 6:
                break

            analysis[option-1]()
        

    def basic_data_inspection(self) -> None:
        """Allows user to inspect the data and generate summary statistics of the given dataframe.
        Provides option for the user to select different data inspection methods.

        Raises:
            ValueError: If the option is not a valid selection
        """

        inspector = DataInspector(None)
        
        while True:

            print('\nEnter Option for Data Inspection\n')
            print('1. Data Types\n2. Summary Statistics\n3. Back')
            try:
                option = int(input())
                if option > 3 or option < 1:
                    raise ValueError
            except ValueError:
                print('Not a valid option, Please re-enter')
                continue

            if option == 3:
                break

            analysis = DataTypesInspection() if option == 1 else SummaryStatsInspection()

            inspector.set_inspection(analysis)
            inspector.execute_inspection(self.df)

    def missing_values_analysis(self) -> None:
        """Performs the missing value analysis.
        """

        MissingValueAnalyser().analyse(self.df)

    def univariate_analysis(self) -> None:
        """Performs the univariate analysis based on user input.

        Raises:
            ValueError: If the option is not a valid selction
        """
        
        analyser = UnivariateAnalyser(None)

        while True:
            print('\nEnter Option\n')
            print('1. Numerical \n2. Categorical \n3. Back')
            try:
                option = int(input())
                if option > 3 or option < 1:
                    raise ValueError
            except ValueError:
                print('Not a valid option, Please re-enter')
                continue

            if option == 3:
                break

            analysis = NumericalUnivariateAnalysis() if option == 1 else CategoricalUnivariateAnalysis()

            print('\nEnter Feature Name\n')
            feature = input()

            analyser.set_analysis(analysis)
            analyser.execute_analysis(self.df, feature)

    def bivariate_analysis(self) -> None:
        """Performs bivariate analysis based on user input

        Raises:
            ValueError: If the option is not a valid selection.
        """

        analyser = BivariateAnalyser(None)


        while True:

            print('\nEnter Option\n')
            print('1. Numerical vs Numerical\n2. Categorical vs Numerical\n3. Back')
            try:
                option = int(input())
                if option > 3 or option < 1:
                    raise ValueError
            except ValueError:
                print('Not a valid option, Please re-enter')
                continue

            if option == 3:
                break

            feature1 = input('\nEnter First Feature\n')
            feature2 = input('\nEnter Second Feature\n')

            analysis = NumericalvsNumeriacalAnalysis() if option == 1 else CategoricalvsNumericalAnalysis()
            analyser.set_analysis(analysis)
            analyser.execute_analysis(self.df, feature1, feature2)

    def multivariate_analysis(self) -> None:
        """Performs the multivariate analysis on the given dataframe based on the features given by user.
        """

        print('\nEnter the names of features to be analysed separated by comma')
        features = input().split(',')

        MultiVariateAnalyser().analyse(self.df[features])




if __name__ == '__main__':

    df = pd.read_csv('./extracted/AmesHousing.csv')
    eda = EDA(df)

    eda.start_analysis()