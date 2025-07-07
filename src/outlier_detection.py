import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
 
# Setup logging configuration
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract class for defing the template for outlier detection
class OutlierDetection(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to detect outliers in the given DataFrame.

        Args:
            df (pd.DataFrame): The dataframe containing features for outlier detection.

        Returns:
            pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        pass


# Z Score Outlier Detection
#---------------------------------------------------------------------
# Class to define method to detect outliers using z score
class ZScoreOutlierDetection(OutlierDetection):
    def __init__(self, threshold: float= 3.0):
        """Initialize ZscoreOutlierDetection with a specific threshold. The feature with z score above 
        this threshold is treated as an outlier.

        Args:
            threshold (float, optional): Threshold to determine outliers. Defaults to 3.0.
        """
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies Outlier Detection using Z Score Method.

        Args:
            df (pd.DataFrame): The dataframe containing features for outlier detection.

        Returns:
            pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        
        logging.info(f'Detecting outliers using Z-Score Method with threshold : {self.threshold}')
        z_score = np.abs((df - df.mean()) / df.std())
        outliers = z_score > self.threshold
        logging.info('Outlier Detection Completed')

        return outliers
    
# IQR Outlier Detection
#---------------------------------------------------------------------
# Class to define method to detect outliers using IQR method
class IQROutlierDetection(OutlierDetection):

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies Outlier Detection using IQR Method.

        Args:
            df (pd.DataFrame): The dataframe containing features for outlier detection.

        Returns:
            pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        
        logging.info(f'Detecting outliers using IQR Method')
        
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 -q1
        outliers = (df < (q1 - 1.5*iqr)) | (df > (q3 + 1.5*iqr))

        logging.info('Outlier Detection Completed')

        return outliers
    

# Class to switch between different outlier detection methods
class OutlierDetector:
    def __init__(self, detection_method: OutlierDetection):
        """Initializes outlier detector with a specific outlier detection method

        Args:
            method (OutlierDetection): The outlier detection method to be used.
        """
        self.detection_method = detection_method

    def set_method(self, detection_method: OutlierDetection) -> None:
        """Sets a new method for detecting outlier

        Args:
            method (OutlierDetection): The new outlier detection method.
        """
        self.detection_method = detection_method

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detects the outliers using the current detection method.

        Args:
            df (pd.DataFrame): The dataframe containing features for outlier detection.

        Returns:
            pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        return self.detection_method.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method: str = 'remove') -> pd.DataFrame:
        """Detects the outlier using the current detection method and then handles the outliers using the specified method.

        Args:
            df (pd.DataFrame): The dataframe containing features for handling outlier .
            method (str, optional): Method to handle the detected outliers. Either remove or cap. Defaults to 'remove'.

        Returns:
            pd.DataFrame: The dataframe after handling the outliers.
        """

        outliers = self.detect_outliers(df)

        if method == 'remove':
            logging.info('Removing Outliers from the dataframe')
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == 'cap':
            logging.info('Capping the outliers in the dataset')
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            raise ValueError(f'Unknown Outlier Handling Method: {method}')
        
        logging.info('Outlier Handling Completed')
        return df_cleaned
    

# example usage
if __name__ == '__main__':
    df = pd.read_csv('./extracted/AmesHousing.csv')
    df = df.select_dtypes(include=[np.number]).dropna()

    detector = OutlierDetector(ZScoreOutlierDetection())
    df_zscore_cleaned = detector.handle_outliers(df)

    detector.set_method(IQROutlierDetection())
    df_iqr_cleaned = detector.handle_outliers(df, method='cap')

    