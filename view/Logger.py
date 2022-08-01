import yaml


class Logger:

    def __init__(self):
        pass

    @staticmethod
    def info(msg: str, **kwargs):
        print(msg, **kwargs)

    @staticmethod
    def print_dict(config: dict) -> None:
        print(yaml.dump(config, default_flow_style=False))

    @staticmethod
    def print_sep(sep: str = '#', length: int = 80) -> None:
        print(sep * length)

    @staticmethod
    def print_title(title: str) -> None:
        Logger.print_sep()
        print('\n' + title.upper() + '\n')
