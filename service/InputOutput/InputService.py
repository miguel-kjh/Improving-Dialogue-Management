import os
from abc import abstractmethod
from abc import ABCMeta


class InputService(metaclass=ABCMeta):

    def _check_path(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'File {path} not found')

    @abstractmethod
    def load(self, path: str) -> object:
        raise NotImplementedError
