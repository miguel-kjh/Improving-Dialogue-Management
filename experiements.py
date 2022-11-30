import subprocess
from view.Logger import Logger

ERRORS = [0, 0.2, 0.6, 0.8, 1]
PYTHON_CMD = 'py'
MAIN_PROGRAM = '.\main.py'
OPTIONS = '-m'
MODELS = [
    ('ted', 'ted'),
    ('ted', 'red'),
    ('dia', 'mc'),
    ('dia', 'md'),
    ('dia', 'seq'),
    ('dia', 'pedp'),
]
DATASET_SYNTHETIC = 'dataset=synthetic'


def check_if_the_relation_of_errors_and_metrics_are_lineal(name_dataset='simple', epochs=10):
    Logger.print_title(f'Check if the relation of errors and metrics are lineal')
    for error in ERRORS:
        Logger.info(f'Error: {error}')
        for state, model in MODELS:
            subprocess.run(
                [
                    PYTHON_CMD,
                    MAIN_PROGRAM,
                    OPTIONS,
                    DATASET_SYNTHETIC,
                    f'dataset.name={name_dataset}_{error}',
                    f'model={model}',
                    f'state={state}',
                    f'model.epochs={epochs}',
                ]
            )


def main():
    check_if_the_relation_of_errors_and_metrics_are_lineal()


if __name__ == '__main__':
    main()
