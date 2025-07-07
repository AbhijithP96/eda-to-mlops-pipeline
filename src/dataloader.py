import os
import logging
import zipfile
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Data Loader
# --------------------------------------------------
# This class defines a common interface for loading data.
class DataLoader(ABC):
    @abstractmethod
    def get_data(self, file_path: str) -> pd.DataFrame:
        """Abstract Method to load data from a given file

        Args:
            file_path (str): Path to the file

        Returns:
            pd.DataFrame: Pandas Dataframe
        """
        pass

# Zip File Data Loader
# --------------------------------------------------
# This class defines method to load data from a zip file
class ZipDataLoader(DataLoader):
    def get_data(self, file_path: str) -> pd.DataFrame:
        """Extracts the zip file and returns the data as pandas dataframe
        """
        
        # check for valid zip files
        if not file_path.endswith('.zip'):
            raise ValueError('The given file is not a valid zip file')
        
        # extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall('extracted')

        # read all csv files
        csv_files = [f for f in os.listdir('extracted') if f.endswith('csv')]

        # raise error if no csv files found
        if len(csv_files) == 0:
            raise FileNotFoundError('No csv files found in given zip file')
        
        # raise error if more than one csv found
        if len(csv_files) > 1:
            raise ValueError('More than one csv files found. Please specify one.')
        
        # read the csv and load the data
        df = pd.read_csv(os.path.join('extracted', csv_files[0]))
        logging.info(f"Data read from {file_path.split('/')[-1]}")

        # return the Dataframe
        return df
        
# CSV File Data Loader
# --------------------------------------------------
# This class defines method to load data from a csv file
class csvDataLoader(DataLoader):
    def get_data(self, file_path: str) -> pd.DataFrame:
        """Read the given csv and loads the data as Pandas datafram
        """

        # check for valid zip files
        if not file_path.endswith('csv'):
            raise ValueError('Not a valid csv file')
        
        # read the csv and load the data
        df = pd.read_csv(file_path)
        logging.info(f"Data read from {file_path.split('/')[-1]}")

        # return the Dataframe
        return df
    
# Data Loader Engine
# --------------------------------------------------
# This class allows to use different data loaders
class DataLoaderEngine:
    @staticmethod
    def get_data_loader(file_extension: str) -> DataLoader:
        """Returns the appropriate Data Loader for the given file extension

        Args:
            file_extension (str): File Extension, either zip or csv

        Raises:
            ValueError: When the given file extension has no data loader 

        Returns:
            DataLoader
        """
        if file_extension == 'zip':
            return ZipDataLoader()
        elif file_extension == 'csv':
            return csvDataLoader()
        else:
            raise ValueError(f'No Data Loader Instance exist for file extension: {file_extension}')


# example usage
if __name__ == '__main__':
    #file_path = './data/archive.zip'
    file_path = './extracted/AmesHousing.csv'

    loader = DataLoaderEngine.get_data_loader('csv')

    df = loader.get_data(file_path)

    print(df.head())