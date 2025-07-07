import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# Setup logging configuration
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract class for defing the template for feature engineering
class FeatureEngineering(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to apply feature engineering transformation to the given dataframe.

        Args:
            df (pd.DataFrame): The dataframe with features to trabsform

        Returns:
            pd.DataFrame: The dtatframe with transformed features.
        """
        pass


# Log Transformation
#----------------------------------------------------------------------
# Class to define the methods to apply log transformation to the selected features
class LogTransformation(FeatureEngineering):
    def __init__(self, features: list):
        """Initializes the LogTransforamtion with specific features to transform.

        Args:
            features (list): List of features in the dataframe to transform.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies log transformation to the selected features in the dataframe

        Args:
            df (pd.DataFrame): The dataframe containing features to transform

        Returns:
            pd.DataFrame: The dataframe with transformed features
        """

        logging.info(f'Log Transformation on the features: {self.features}')
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature]) #log1p handles log(0)

        logging.info('Log Transformation Completed')
        return df_transformed
    
# Standard Scaler
#-----------------------------------------------------------------------
# Class to define method to apply standard scaler to the specific features
class StandardScalerTransformation(FeatureEngineering):
    def __init__(self, features: list):
        """Initializes the StandardScalerTranformation with the specific features to transform

        Args:
            features (list): List of features in the dataframe to transform
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies Standard Scaler to the selected features in the given dataframe

        Args:
            df (pd.DataFrame): The dataframe containing features to transform

        Returns:
            pd.DataFrame: The dataframe with transformed features
        """

        logging.info(f'Standard Scaler Transformaton on features: {self.features}')
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard Scaler Transformation Completed")

        return df_transformed


# MinMax Scaler 
# ---------------------------------------------------------------
# Class to define method to apply min max scaling to specific features
class MinMaxScalerTransformation(FeatureEngineering):
    def __init__(self, features: list, feature_range: tuple = (0, 1)):
        """Initializes the MinMaxScalerTranformation with the specific features to transform

        Args:
            features (list): List of features in the dataframe to transform
            feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies MinMax Scaler to the selected features in the given dataframe

        Args:
            df (pd.DataFrame): The dataframe containing features to transform

        Returns:
            pd.DataFrame: The dataframe with transformed features
        """

        logging.info(f'MinMax Scaler Transformaton on features: {self.features} in range: {self.scaler.feature_range}')
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("MinMax Scaler Transformation Completed")

        return df_transformed
    

# One Hot Encoding
#------------------------------------------------------------
# Class to define method to apply one hot encoding to specific cateogorical features
class OneHotEncoding(FeatureEngineering):
    def __init__(self, features: list):
        """Initializes the OneHotEncoding with the specific categorical features to transform

        Args:
            features (list): List of categorical features in the dataframe to transform
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop='first')

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies One hot encoding to the selected categorical features in the given dataframe

        Args:
            df (pd.DataFrame): The dataframe containing features to transform

        Returns:
            pd.DataFrame: The dataframe with transformed features
        """

        logging.info(f'One Hot Encoding on features: {self.features}')
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features)
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One Hot Encoding Completed")

        return df_transformed
    

# Feature Engineer
#----------------------------------------------------
# Class to switch between different feature engineering
class FeatureEngineer:
    def __init__(self, engineering: FeatureEngineering):
        """Initializes the FeatureEngineer with a specific feature engineering strategy.

        Args:
            engineering (FeatureEngineering): The strategy to be used for feature engineering.
        """

        self.engineering = engineering

    def set_strategy(self, engineering: FeatureEngineering) -> None:
        """
        Sets a new strategy for the FeatureEngineer.

        Args:
        engineering (FeatureEngineering): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self.engineering = engineering

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Args:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy.")
        return self.engineering.apply_transformation(df)
    
# example usage
if __name__ == '__main__':
    df = pd.read_csv('./extracted/AmesHousing.csv')

    features = ['SalePrice']
    cat = ['Neighborhood']

    engineer = FeatureEngineer(LogTransformation(features))
    df=engineer.apply_feature_engineering(df)

    engineer.set_strategy(StandardScalerTransformation(features))
    df=engineer.apply_feature_engineering(df)

    engineer.set_strategy(MinMaxScalerTransformation(features))
    df=engineer.apply_feature_engineering(df)

    engineer.set_strategy(OneHotEncoding(cat))
    engineer.apply_feature_engineering(df)
