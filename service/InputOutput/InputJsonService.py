import json

from .InputService import InputService


class InputJsonService(InputService):

    def __init__(self):
        super().__init__()
        self._modes = 'r'

    def load(self, path: str) -> dict:
        self._check_path(path)

        with open(path, self._modes) as read_file:
            data = json.load(read_file)

        return data
