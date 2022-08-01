import os
from abc import abstractmethod
from abc import ABCMeta


class OutputService(metaclass=ABCMeta):

    @abstractmethod
    def save(self, data: object, path: str) -> None:
        raise NotImplementedError
