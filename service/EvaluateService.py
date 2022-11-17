import numpy as np
import pandas as pd
import wandb
from typing import List

from tqdm import tqdm

from service.Pipeline import Pipeline
import pytorch_lightning as pl

from view.Logger import Logger
from view.Metrics import Metrics


class EvaluateService(Pipeline):

    def __init__(self, data: pl.LightningDataModule, actions: List[str]):
        super().__init__()
        self.data = data
        self.dataset = data.dataset
        self.actions = list(sorted(actions))

    def _transform_binary_embeddings(self, indexes: list) -> tuple:
        intentions = []
        entities = []
        slots = []
        pre_actions = []
        dialogue_id = []

        for record_index in tqdm(indexes, desc='Transforming embeddings'):
            intentions.append(self.dataset['Intention'][record_index])
            slots.append(self.dataset['Slots'][record_index])
            pre_actions.append(self.dataset['Prev_action'][record_index])
            entities.append(self.dataset['Entities'][record_index])
            dialogue_id.append(self.dataset['Dialogue_ID'][record_index])

        return intentions, entities, slots, pre_actions, dialogue_id

    def _convert_actions(self, actions: List[str]):
        if not actions:
            return []
        return [self.actions[number_action] for number_action in actions]

    def _get_number_of_actions(self, actions: List[List[int]]) -> List[List[int]]:
        result = [list(np.where(x == 1)[0]) for x in actions]
        result = [self._convert_actions(x) for x in result]
        return result

    def _update_test_results(self, test_results: dict):
        intentions, entities, slots, pre_actions, dialogue_id = self._transform_binary_embeddings(test_results['Index'])

        test_results['Dialogue_ID'] = dialogue_id
        test_results['Intentions'] = intentions
        test_results['Entities'] = entities
        test_results['Pre_actions'] = pre_actions
        test_results['Slots'] = slots

        if 'Ranking' in test_results:
            test_results['Ranking'] = [self._convert_actions(x) for x in test_results['Ranking']]
            test_results['Labels'] = self._convert_actions(test_results['Labels'])
            test_results['Predictions'] = self._convert_actions(test_results['Predictions'])
        else:
            test_results['Labels'] = self._get_number_of_actions(test_results['Labels'])
            test_results['Predictions'] = self._get_number_of_actions(test_results['Predictions'])

    def _create_confusion_matrix(self, test_results: pd.DataFrame):
        labels = np.hstack(test_results['Labels'].values).tolist()
        predictions = np.hstack(test_results['Predictions'].values).tolist()
        if predictions:
            try:
                return Metrics.plot_confusion_matrix(
                    labels,
                    predictions,
                    title='Confusion matrix',
                )
            except ValueError:
                Logger.error('Confusion matrix cannot be created')
                return None
        else:
            Logger.warning('No predictions were made')
            return None

    def run(self, data: object = None) -> object:

        assert isinstance(data, pl.Trainer), "Data must be of type Trainer"

        test = data.test(datamodule=self.data)
        test_results = data.model.test_results
        self._update_test_results(test_results)
        test = pd.DataFrame(test)
        test_results = pd.DataFrame(test_results)
        cm = self._create_confusion_matrix(test_results)

        wandb.finish()

        return test_results, test, cm
