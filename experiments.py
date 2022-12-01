import copy
import re
import subprocess
import os
from tqdm import tqdm

from view.Logger import Logger
import pandas as pd

ERRORS = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
PYTHON_CMD = 'py'
MAIN_PROGRAM = '.\main.py'
OPTIONS = '-m'
MODELS = [
    ('ted', 'ted'),
    ('ted', 'red'),
    ('dia', 'md'),
    ('dia', 'mc'),
    ('dia', 'seq'),
    ('pedp', 'pedp'),
]
DATASET_SYNTHETIC = 'dataset=synthetic'
EPOCHS = 1

PRINCIPAL_FOLDER = 'experiments'
FIRST_EXPERIMENT = os.path.join(PRINCIPAL_FOLDER, 'check_if_the_relation_of_errors_and_metrics_are_lineal')


def create_folder():
    # create folder experiments
    if not os.path.exists(PRINCIPAL_FOLDER):
        os.mkdir(PRINCIPAL_FOLDER)
    # create folder experiments/check_if_the_relation_of_errors_and_metrics_are_lineal
    if not os.path.exists(FIRST_EXPERIMENT):
        os.mkdir(FIRST_EXPERIMENT)
    else:
        for file in os.listdir(FIRST_EXPERIMENT):
            os.remove(os.path.join(FIRST_EXPERIMENT, file))
    Logger.print_title('Folder created')


def check_if_the_relation_of_errors_and_metrics_are_lineal(name_dataset='simple', epochs=EPOCHS):
    results = {
        'comparative': {
            'error': ERRORS,
        },
    }
    for state, model in tqdm(MODELS, desc='Check if the relation of errors and metrics are lineal'):
        results[model] = {
            'error': [],
            'test_accuracy': [],
            'test_f1': [],
            'test_precision': [],
            'test_recall': [],
        }
        results['comparative'][model] = []
        for error in ERRORS:
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
            regex = r'test_recall: \d.\d+|test_accuracy: \d.\d+|test_precision: \d.\d+|test_f1: \d.\d+'
            metrics = re.findall(regex, out.decode('utf-8'))
            metrics = [m.split(':') for m in metrics]
            for metric in metrics:
                results[model][metric[0].strip()].append(float(metric[1].strip()))
            results[model]['error'].append(error)
        results['comparative'][model] = copy.deepcopy(results[model]['test_f1'])

    # save in one excel file with one sheet per model
    with pd.ExcelWriter(os.path.join(FIRST_EXPERIMENT, 'results.xlsx')) as writer:
        for model, result in results.items():
            df = pd.DataFrame(result)
            df.to_excel(writer, sheet_name=model, index=False)
    Logger.print_title('Results saved')


def main():
    create_folder()
    check_if_the_relation_of_errors_and_metrics_are_lineal()


if __name__ == '__main__':
    main()
