import pandas as pd
from src.dataloader import DataLoaderEngine
from zenml import step

@step
def data_loader_step(file_path: str) -> pd.DataFrame:

    # get extension from the file path
    extension = file_path[-3:]

    # get the appropriate loader for the given extension
    loader = DataLoaderEngine().get_data_loader(extension)

    # get the data frame
    df = loader.get_data(file_path)

    return df


if __name__ == '__main__':
    data_loader_step('./data/archive.zip')