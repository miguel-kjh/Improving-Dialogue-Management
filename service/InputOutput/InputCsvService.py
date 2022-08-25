import os

import pandas as pd

from .InputService import InputService


class InputCsvService(InputService):

    def __init__(self, columns: list = None):
        super().__init__()

        if not columns:
            columns = []

        self._columns = columns
        self._sep = ';'

    def load(self, path: str) -> pd.DataFrame:

        self._check_path(path)

        if self._columns:
            return pd.read_csv(
                path,
                usecols=self._columns,
                sep=self._sep
            )
        else:
            return pd.read_csv(
                path,
                sep=self._sep
            )


