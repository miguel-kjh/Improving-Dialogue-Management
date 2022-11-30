import re
import subprocess

from tqdm import tqdm


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
    ('pedp', 'pedp'),
]
DATASET_SYNTHETIC = 'dataset=synthetic'
EPOCHS = 1


def check_if_the_relation_of_errors_and_metrics_are_lineal(name_dataset='simple', epochs=EPOCHS):
    for error in tqdm(ERRORS, desc='Check if the relation of errors and metrics are lineal'):
        for state, model in MODELS:
            process = subprocess.Popen(
                [
                    PYTHON_CMD,
                    MAIN_PROGRAM,
                    OPTIONS,
                    DATASET_SYNTHETIC,
                    f'dataset.name={name_dataset}_{error}',
                    f'model={model}',
                    f'state={state}',
                    f'model.epochs={epochs}',
                ],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            out, _ = process.communicate()
            # get metrcis from out
            metrics = re.findall(r'(?<=\n)\d+\.\d+', out.decode('utf-8'))
            print(metrics)
            exit()


def main():
    check_if_the_relation_of_errors_and_metrics_are_lineal()


if __name__ == '__main__':
    main()
