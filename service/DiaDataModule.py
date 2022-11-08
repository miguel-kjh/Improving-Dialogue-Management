from abc import ABC
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader


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
        state = np.array(row['State'].tolist())
        action = np.array(row['Set_Label'].tolist())
        last_pos = [np.where(action != 0)[0][-1] + 1]
        return state, action, last_pos


class DiaDataModule(pl.LightningDataModule, ABC):

    def __init__(self, df: pd.DataFrame, batch_size: int = 32):
        super(DiaDataModule, self).__init__()
        self.df_train = None
        self.df_dev = None
        self.df_test = None
        self.df = df
        self.batch_size = batch_size

    def _prepare_data(self, type_data: str) -> DiaDataset:
        df = self.df[self.df['set'] == type_data]
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