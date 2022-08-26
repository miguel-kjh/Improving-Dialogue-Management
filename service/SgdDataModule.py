import os
import pickle
from abc import ABC

import torch
from typing import Optional, List
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

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


class SgdDataModule(pl.LightningDataModule, ABC):

    def __init__(
            self,
            train: np.array,
            label_train: np.array,
            set_labels_train: np.array,
            val: np.array,
            label_val: np.array,
            set_labels_val: np.array,
            test: np.array,
            label_test: np.array,
            set_labels_test: np.array,
            train_indexes,
            validation_indexes,
            test_indexes,
            batch_size: int = 32
    ):
        super().__init__()
        self.sgd_test = None
        self.sgd_val = None
        self.sgd_train = None
        self.batch_size = batch_size
        self.train = train
        self.label_train = label_train
        self.set_labels_train = set_labels_train
        self.val = val
        self.label_val = label_val
        self.set_labels_val = set_labels_val
        self.test = test
        self.label_test = label_test
        self.set_labels_test = set_labels_test
        self.train_indexes = train_indexes
        self.validation_indexes = validation_indexes
        self.test_indexes = test_indexes

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

    def train_dataloader(self):
        sampler = StratifiedSampler(class_vector=self.sgd_train.labels, batch_size=self.batch_size)
        return DataLoader(self.sgd_train, batch_size=self.batch_size, sampler=sampler)

    def val_dataloader(self):
        sampler = StratifiedSampler(class_vector=self.sgd_val.labels, batch_size=self.batch_size)
        return DataLoader(self.sgd_val, batch_size=self.batch_size, sampler=sampler)

    def test_dataloader(self):
        #sampler = SequentialSampler(len(self.sgd_val.labels))
        #sampler = StratifiedSampler(class_vector=self.sgd_test.labels, batch_size=self.batch_size)
        return DataLoader(self.sgd_test, batch_size=self.batch_size)
