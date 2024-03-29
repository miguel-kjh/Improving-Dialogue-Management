import copy
from abc import ABC

import pandas as pd
import torch

from models.datamodule.DiaDataModule import DiaDataModule


class PepdDataset:

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
        s_pos = torch.tensor([row['Last_position']], dtype=torch.long).T
        actions = torch.tensor(row['Real_label'], dtype=torch.long)
        try:
            next_state = torch.tensor(
                self.df.iloc[index + 1]['State'].tolist()
            )
            a_target_full = torch.tensor(
                torch.tensor(
                    self.df.iloc[index + 1]['Set_Label'].tolist(),
                    dtype=torch.long
                )
            )
            last_pos = torch.tensor([self.df.iloc[index + 1]['Last_position']], dtype=torch.long).T
        except IndexError:
            next_state = torch.zeros_like(state)
            a_target_full = torch.zeros_like(a_target_gold)
            last_pos = torch.zeros_like(s_pos)

        return state, next_state, s_pos, a_target_gold, last_pos, a_target_full, actions, index


class PepdDataModule(DiaDataModule, ABC):

    def _prepare_data(self, type_data: str) -> PepdDataset:
        df = self.df[self.df['Type'] == type_data]
        return PepdDataset(df)
