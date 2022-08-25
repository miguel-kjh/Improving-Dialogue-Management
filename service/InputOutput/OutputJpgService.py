from matplotlib import pyplot as plt
from plotly.graph_objs import Figure

from service.InputOutput.OutputService import OutputService


class OutputJpgService(OutputService):

    def __init__(self):
        super().__init__()

    def save(self, data: object, path: str) -> None:

        assert isinstance(data, Figure)

        data.write_image(path)

