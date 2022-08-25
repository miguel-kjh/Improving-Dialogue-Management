import pickle

from .OutputService import OutputService


class OutputPklService(OutputService):
    def __init__(self):
        super().__init__()
        self._mode = 'wb'

    def save(self, data: object, path: str) -> None:

        with open(path, self._mode) as file:
            pickle.dump(data, file)
