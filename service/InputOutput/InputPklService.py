import pickle

from .InputService import InputService


class InputPklService(InputService):

    def __init__(self):
        super().__init__()
        self._modes = 'rb'

    def load(self, path: str) -> object:

        self._check_path(path)

        with open(path, self._modes) as read_file:
            data = pickle.load(read_file)

        return data
