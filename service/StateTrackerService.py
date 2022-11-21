import pandas as pd

from models.datamodule.DiaDataModule import DiaDataModule
from models.datamodule.PepdDataModule import PepdDataModule
from models.datamodule.TedDataModule import TedDataModule
from models.dialogue_state_tracker.DiaStateCreator import DiaStateCreator
from models.dialogue_state_tracker.TedStateCreator import TedStateCreator
from service.Pipeline import Pipeline


class StateTrackerService(Pipeline):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.state_creator = {
            'ted': TedStateCreator,
            'dia': DiaStateCreator,
            'pepd': DiaStateCreator
        }
        self.data_module = {
            'ted': TedDataModule,
            'dia': DiaDataModule,
            'pepd': PepdDataModule
        }

    def run(self, data: object = None) -> object:
        assert isinstance(data, pd.DataFrame), 'data must be a pandas DataFrame'
        if self.config['state']['name'] == 'ted':
            state_creator = TedStateCreator(
                data,
                self.config['state']['intention'],
                self.config['state']['action'],
                self.config['state']['max_history'],
            )
            df_state = state_creator.create_dataset()

            return TedDataModule(
                df_state,
                self.config['model']['batch_size']
            )
        elif self.config['state']['name'] == 'dia':
            state_creator = DiaStateCreator(
                data,
                self.config['state']['intention'],
                self.config['state']['action']
            )
            df_state = state_creator.create_dataset()
            return DiaDataModule(
                df_state,
                self.config['model']['batch_size']
            )
        else:
            state_creator = DiaStateCreator(
                data,
                self.config['state']['intention'],
                self.config['state']['action']
            )
            df_state = state_creator.create_dataset()
            return PepdDataModule(
                df_state,
                self.config['model']['batch_size']
            )
