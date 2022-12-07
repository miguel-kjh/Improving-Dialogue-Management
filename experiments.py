import copy
import re
import subprocess
import os

import numpy as np
from tqdm import tqdm
from typing import List

from view.Logger import Logger
import pandas as pd

PROP = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
DATASETS = ['ds1', 'ds2', 'ds3']
PYTHON_CMD = 'py'
MAIN_PROGRAM = '.\main.py'
OPTIONS = '-m'
MODELS = [
    ('ted', 'ted'),
    ('ted', 'red'),
    # ('dia', 'md'),
    # ('dia', 'mc'),
    # ('dia', 'seq'),
    # ('pedp', 'pedp'),
]
DATASET_SYNTHETIC = 'dataset=synthetic'
EPOCHS = 10
GPUs = 0

PRINCIPAL_FOLDER = 'experiments'

THIRD_EXPERIMENT = os.path.join(PRINCIPAL_FOLDER, 'check_if_the_problem_is_simple_or_complex')
FIRST_EXPERIMENT = os.path.join(PRINCIPAL_FOLDER, 'check_if_the_relation_of_errors_and_metrics_are_lineal')
SECOND_EXPERIMENT = os.path.join(PRINCIPAL_FOLDER, 'experiments_to_events')


def create_folder(experiments: List[str] = None):
    if not experiments:
        experiments = [THIRD_EXPERIMENT, FIRST_EXPERIMENT, SECOND_EXPERIMENT]

    for experiment in experiments:
        if not os.path.exists(experiment):
            os.makedirs(experiment)
        else:
            for file in os.listdir(experiment):
                os.remove(os.path.join(experiment, file))

    Logger.print_title('Folder created')


def check_if_the_problem_is_simple_or_complex():
    results = {}
    for dataset in tqdm(DATASETS, desc='Check if the problem is simple or complex'):
        results[dataset] = {
            'model': [],
            'test_accuracy': [],
            'test_f1': [],
            'test_precision': [],
            'test_recall': [],
        }
        for state, model in MODELS:
            process = subprocess.Popen(
                [
                    PYTHON_CMD,
                    MAIN_PROGRAM,
                    OPTIONS,
                    DATASET_SYNTHETIC,
                    f'dataset.name={dataset}',
                    f'model={model}',
                    f'state={state}',
                    f'model.epochs={EPOCHS}',
                    f'resources.gpus={GPUs}',
                ],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            out, _ = process.communicate()
            regex = r'test_recall: \d.\d+|test_accuracy: \d.\d+|test_precision: \d.\d+|test_f1: \d.\d+'
            metrics = re.findall(regex, out.decode('utf-8'))
            metrics = [m.split(':') for m in metrics]
            for metric in metrics:
                results[dataset][metric[0].strip()].append(float(metric[1].strip()))
            results[dataset]['model'].append(model)

    # save in one excel file with one sheet per dataset
    with pd.ExcelWriter(os.path.join(THIRD_EXPERIMENT, 'results.xlsx')) as writer:
        for dataset, result in results.items():
            df = pd.DataFrame(result)
            df.to_excel(writer, sheet_name=dataset, index=False)
    Logger.print_title('Results saved')



def check_if_the_relation_of_errors_and_metrics_are_lineal(name_dataset='simple', epochs=EPOCHS):
    results = {
        'comparative': {
            'error': PROP,
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
        for error in PROP:
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


def experiments_to_events(events: List[str] = None):
    if not events:
        events = ['chit_chat', 'change_idea', 'change_domain', 'all_events']

    results = {}
    for event in tqdm(events, desc='Experiments to events'):
        results[event] = {
            'model': [],
            'error': [],
            'test_accuracy': [],
            'test_f1': [],
            'test_precision': [],
            'test_recall': [],
        }
        for state, model in MODELS:
            for error in PROP:
                results[event]['model'].append(model)
                results[event]['error'].append(error)
                process = subprocess.Popen(
                    [
                        PYTHON_CMD,
                        MAIN_PROGRAM,
                        OPTIONS,
                        DATASET_SYNTHETIC,
                        f'dataset.name=simple_{event}_{error}',
                        f'model={model}',
                        f'state={state}',
                        f'model.epochs={EPOCHS}',
                        f'resources.gpus=0',
                    ],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
                out, _ = process.communicate()
                regex = r'test_recall: \d.\d+|test_accuracy: \d.\d+|test_precision: \d.\d+|test_f1: \d.\d+'
                metrics = re.findall(regex, out.decode('utf-8'))
                metrics = [m.split(':') for m in metrics]
                for metric in metrics:
                    results[event][metric[0].strip()].append(float(metric[1].strip()))

    # save in one excel file with one sheet per event
    with pd.ExcelWriter(os.path.join(SECOND_EXPERIMENT, 'results.xlsx')) as writer:
        for event, result in results.items():
            df = pd.DataFrame(result)
            df.to_excel(writer, sheet_name=event, index=False)
    Logger.print_title('Results saved')


def main():
    create_folder()
    check_if_the_problem_is_simple_or_complex()
    #experiments_to_events()
    #check_if_the_relation_of_errors_and_metrics_are_lineal()


if __name__ == '__main__':
    main()
