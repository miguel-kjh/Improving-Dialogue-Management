import os
import json
import pickle
import pandas as pd


class SaveLoadService:

    def __init__(self) -> None:
        self.folders = {
            'raw': os.path.join('data', 'raw'),
            'int': os.path.join('data', 'int'),
            'res': os.path.join('data', 'res'),
            'clustering': os.path.join('data', 'clustering')
        }

    def load_json(self, data_file: str) -> dict:
        if os.path.isfile(data_file):
            with open(data_file, 'r') as read_file:
                data = json.load(read_file)
                return data

    def load_pkl(self, data_file: str):
        if os.path.isfile(data_file):
            with open(data_file, 'r') as read_file:
                data = pickle.load(read_file)
            return data

    def save_json(self, json_data, filename: str, folder: str = 'res'):

        with open(os.path.join(self.folders[folder], filename), 'w') as fp:
            json.dump(json_data, fp)

    def save_pkl(self, object, filename: str, folder: str = 'res'):

        with open(os.path.join(self.folders[folder], filename), 'wb') as f:
            pickle.dump(object, f)

    def save_dataframe(self, df, filename: str, folder: str = 'res'):
        df.to_csv(os.path.join(self.folders[folder], filename), sep=';')

    def save_to_csv(self, filename: str, df: pd.DataFrame, folder: str = 'res'):
        df.to_csv(os.path.join(self.folders[folder], filename), index=False, sep=';')
