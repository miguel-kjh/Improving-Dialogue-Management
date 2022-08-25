import json

from .OutputService import OutputService


class OutputJsonService(OutputService):
    def __init__(self):
        super().__init__()
        self._mode = 'w'

    def save(self, data: object, path: str) -> None:
        with open(path, self._mode) as file:
            json.dump(data, file)


