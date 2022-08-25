from service.InputOutput.OutputService import OutputService
from plotly.graph_objs import Figure


class OutputHtmlService(OutputService):

    def __init__(self):
        super().__init__()

    def save(self, data: object, path: str) -> None:
        assert isinstance(data, Figure), "data must be a plotly.graph_objs.Figure"

        data.write_html(path)
