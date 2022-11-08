from abc import abstractmethod
from abc import ABCMeta

from service.InputOutput.InputCsvService import InputCsvService
from service.InputOutput.OutputCsvService import OutputCsvService
from service.InputOutput.OutputHtmlService import OutputHtmlService
from service.InputOutput.OutputJsonService import OutputJsonService
from service.InputOutput.InputJsonService import InputJsonService
from service.InputOutput.InputPklService import InputPklService
from service.InputOutput.OutputPklService import OutputPklService
from service.InputOutput.OutputJpgService import OutputJpgService


class Pipeline(metaclass=ABCMeta):

    def __init__(self):

        self.input_csv_service = InputCsvService()
        self.output_csv_service = OutputCsvService()

        self.input_json_service = InputJsonService()
        self.output_json_service = OutputJsonService()

        self.input_pkl_service = InputPklService()
        self.output_pkl_service = OutputPklService()

        self.output_jpg_service = OutputJpgService()
        self.output_html_service = OutputHtmlService()

        self.filename = 'SGD_dataset'

        self.priority = 0

    @abstractmethod
    def run(self, data: object = None) -> object:
        raise NotImplementedError

    # compare the priority of the pipeline
    def __lt__(self, other):
        return self.priority < other.priority

