from abc import ABC
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.datamodule.DataModule import DataModule


class DiaDataset:

    def __init__(self, df: pd.DataFrame):
        """Initialization"""
        self.df = df

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.df)

    def __getitem__(self, index):
        """Generates one sample of data"""
        row = self.df.iloc[index]
        state = torch.tensor(row['State'].tolist())
        a_target_gold = torch.tensor(row['Set_Label'].tolist(), dtype=torch.long)
        last_pos = torch.tensor(row['Last_position'], dtype=torch.long)
        actions = torch.tensor(row['Real_label'], dtype=torch.long)
        return state, a_target_gold, last_pos, actions, index


class DiaDataModule(DataModule, ABC):

    def __init__(self, df: pd.DataFrame, batch_size: int = 32):
        super(DiaDataModule, self).__init__(df)
        self.df_train = None
        self.df_dev = None
        self.df_test = None
        self._num_features = len(self.df['State'].iloc[0])
        self.batch_size = batch_size

    def _prepare_data(self, type_data: str) -> DiaDataset:
        df = self.df[self.df['Type'] == type_data]
        return DiaDataset(df)

    def setup(self, stage: Optional[str] = None):
        self.df_train = self._prepare_data('train')
        self.df_dev = self._prepare_data('dev')
        self.df_test = self._prepare_data('test')

    def train_dataloader(self):
        return DataLoader(self.df_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.df_dev, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.df_test, batch_size=self.batch_size)
