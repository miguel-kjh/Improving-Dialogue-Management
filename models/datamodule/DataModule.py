from abc import ABC

import numpy as np
import pandas as pd
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule, ABC):

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self._num_features = len(self.df['State'].iloc[0])
        self._num_classes = len(self.df['Label'].iloc[0])
        self._classes = sorted(list(set(np.hstack(df['Label']))))

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def classes(self):
        return self._classes

    @property
    def dataset(self):
        return self.df
