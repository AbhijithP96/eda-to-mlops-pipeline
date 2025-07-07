from analysis.EDA import EDA
from src.dataloader import DataLoaderEngine

if __name__ == "__main__":

    file_path = input('Enter path to dataset file (zip or csv)\n')

    loader = DataLoaderEngine().get_data_loader(file_path[-3:])
    df = loader.get_data(file_path)

    analyser = EDA(df)
    analyser.start_analysis()
