import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from models.datamodule.DiaDataModule import DiaDataModule
from models.datamodule.TedDataModule import TedDataModule
from models.dialogue_state_tracker.DiaStateCreator import DiaStateCreator
from models.dialogue_state_tracker.TedStateCreator import TedStateCreator
from service.Pipeline import Pipeline


class StateTrackerService(Pipeline):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def _get_samples_by_dataset(self, df_state: pd.DataFrame, type_data: str) -> tuple:
        action_encoder = LabelEncoder()
        action_encoder.fit(df_state['Label'])
        dataset = df_state[df_state['Type'] == type_data]
        indexes = dataset['Index'].to_list()
        labels = action_encoder.transform(dataset['Label'])
        set_labels = [
            list(set(action_encoder.transform(actions)))
            for actions in dataset['Set_Label']
        ]
        max_len_of_set_labels = max([len(set_of_label) for set_of_label in set_labels])
        set_labels = [
            set_of_label + [-1] * (max_len_of_set_labels - len(set_of_label))
            for set_of_label in set_labels
        ]
        features = np.array(dataset['State'].to_list())

        assert len(indexes) == len(features) == len(labels) == len(set_labels), \
            f"Bab dimension of features: {len(features)} - " \
            f"labels : {len(labels)} - set_labels: {len(set_labels)} " \
            f"index: {len(indexes)}"
        numbers = [len(set_label) for set_label in set_labels]
        assert max(numbers) == min(numbers), "Bab dimension of set_labels"

        return features, labels, set_labels, indexes

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
            train, train_labels, set_train_labels, train_indexes = self._get_samples_by_dataset(df_state, 'train')
            validation, validation_labels, set_validation_labels, validation_indexes = self._get_samples_by_dataset(df_state, 'dev')
            test, test_labels, set_test_labels, test_indexes = self._get_samples_by_dataset(df_state, 'test')

            return TedDataModule(
                train,
                train_labels,
                set_train_labels,
                validation,
                validation_labels,
                set_validation_labels,
                test,
                test_labels,
                set_test_labels,
                train_indexes,
                validation_indexes,
                test_indexes,
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

