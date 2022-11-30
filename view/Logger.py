import pandas as pd
import yaml


class Logger:

    def __init__(self):
        pass

    @staticmethod
    def info(msg: str, **kwargs):
        print(msg, **kwargs)

    @staticmethod
    def print_test(
            test: pd.DataFrame,
            metrics=None,
            decimals: int = 2
    ):

        if metrics is None:
            metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']

        for metric in metrics:
            print(f"{metric}: {round(test[metric].item(), decimals)}")

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

    @staticmethod
    def warning(msg: str, **kwargs):
        print('WARNING: ' + msg, **kwargs)

    @staticmethod
    def error(msg: str, **kwargs):
        print('ERROR: ' + msg, **kwargs)