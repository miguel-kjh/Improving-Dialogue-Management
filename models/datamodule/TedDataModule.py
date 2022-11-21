import os
import pickle
from abc import ABC

import pandas as pd
import torch
from typing import Optional, List
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from models.datamodule.DataModule import DataModule
from models.transformation.StratifiedSampler import StratifiedSampler


def _load_pkl_utils(data_file: str):
    if os.path.isfile(data_file):
        with open(data_file, 'rb') as read_file:
            data = pickle.load(read_file)
        return data


def listToTensor(data: List) -> torch.Tensor:
    return torch.FloatTensor(data)


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, labels, set_labels, indexes_dataset):
        """Initialization"""

        assert len(X) == len(labels)

        self.labels = labels
        self.X = X
        self.set_labels = set_labels
        self.indexes_dataset = indexes_dataset

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""

        if index > len(self.X):
            index = len(self.X) - 1
        y = self.labels[index]
        set_y = self.set_labels[index]
        x = self.X[index]
        real_index = self.indexes_dataset[index]
        return listToTensor(x), y, torch.tensor(set_y).T, real_index


class TedDataModule(DataModule, ABC):

    #TODO: evitar que tenga tantos parámetros, solo el df debería ser suficiente

    def __init__(
            self,
            df: pd.DataFrame,
            batch_size: int = 32
    ):
        super().__init__(df)
        train, train_labels, set_train_labels, train_indexes = self._get_samples_by_dataset(df, 'train')
        validation, validation_labels, set_validation_labels, validation_indexes = self._get_samples_by_dataset(df, 'dev')
        test, test_labels, set_test_labels, test_indexes = self._get_samples_by_dataset(df, 'test')
        self.sgd_test = None
        self.sgd_val = None
        self.sgd_train = None
        self.batch_size = batch_size
        self.train = train
        self.label_train = train_labels
        self.set_labels_train = set_train_labels
        self.val = validation
        self.label_val = validation_labels
        self.set_labels_val = set_validation_labels
        self.test = test
        self.label_test = test_labels
        self.set_labels_test = set_test_labels
        self.train_indexes = train_indexes
        self.validation_indexes = validation_indexes
        self.test_indexes = test_indexes
        self._num_features = train.shape[1]
        self._num_classes = len(np.unique(train_labels))

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

    def setup(self, stage: Optional[str] = None):

        self.sgd_train = Dataset(
            self.train,
            self.label_train,
            self.set_labels_train,
            self.train_indexes
        )
        self.sgd_val = Dataset(
            self.val,
            self.label_val,
            self.set_labels_val,
            self.validation_indexes
        )
        self.sgd_test = Dataset(
            self.test,
            self.label_test,
            self.set_labels_test,
            self.test_indexes
        )

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_features(self):
        return self._num_features

    def train_dataloader(self):
        try:
            sampler = StratifiedSampler(class_vector=self.sgd_train.labels, batch_size=self.batch_size)
            sampler.gen_sample_array()
            return DataLoader(self.sgd_train, batch_size=self.batch_size, sampler=sampler)
        except Exception as e:
            return DataLoader(self.sgd_train, batch_size=self.batch_size)

    def val_dataloader(self):
        try:
            sampler = StratifiedSampler(class_vector=self.sgd_val.labels, batch_size=self.batch_size)
            sampler.gen_sample_array()
            return DataLoader(self.sgd_val, batch_size=self.batch_size, sampler=sampler)
        except Exception as e:
            return DataLoader(self.sgd_val, batch_size=self.batch_size)

    def test_dataloader(self):
        #sampler = SequentialSampler(len(self.sgd_val.labels))
        #sampler = StratifiedSampler(class_vector=self.sgd_test.labels, batch_size=self.batch_size)
        return DataLoader(self.sgd_test, batch_size=self.batch_size)
