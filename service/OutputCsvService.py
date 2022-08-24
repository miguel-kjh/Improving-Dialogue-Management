import pandas as pd

from .OutputService import OutputService


class OutputCsvService(OutputService):

    def __init__(self) -> None:
        super().__init__()
        self._sep = ";"

    def save(self, data: pd.DataFrame, path: str) -> None:

        data.to_csv(
            path,
            sep=self._sep,
            index=False
        )


