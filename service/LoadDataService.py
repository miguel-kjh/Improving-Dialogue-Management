import os

import pandas as pd

from service.InputOutput.MongoDB import MongoDB
from service.Pipeline import Pipeline
from view.Logger import Logger


class LoadDataService(Pipeline):

    def __init__(self, config: dict):
        super().__init__()
        self.mongodb_service = MongoDB(
            config['dataset']['DB_name'],
            config['database'][0]['path']
        )
        name = config['dataset']['name']
        domain = config['dataset']['domain']
        self.dataset_name = f"{name}_{domain}"

    def run(self, data: object = None) -> object:
        Logger.info('Loading data from MongoDB')
        df = self.mongodb_service.load(self.dataset_name)
        return df
