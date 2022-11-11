import pandas as pd
import wandb
from typing import List

from service.Pipeline import Pipeline
import pytorch_lightning as pl


class EvaluateService(Pipeline):

    def __init__(self, data: pl.LightningDataModule, actions: List[str]):
        super().__init__()
        self.data = data
        self.dataset = data.dataset
        self.actions = actions

    def _transform_binary_embeddings(self, indexes: list) -> tuple:
        intentions = []
        slots = []
        pre_actions = []
        dialogue_id = []

        for record_index in indexes:
            intentions.append(self.dataset['Intention'][record_index])
            slots.append(self.dataset['Slots'][record_index])
            pre_actions.append(self.dataset['Prev_action'][record_index])
            dialogue_id.append(self.dataset['Dialogue_ID'][record_index])

        return intentions, slots, pre_actions, dialogue_id

    def _update_test_results(self, test_results: dict):
        intentions, slots, pre_actions, dialogue_id = self._transform_binary_embeddings(test_results['Index'])
        test_results['Dialogue_ID'] = dialogue_id
        test_results['Intentions'] = intentions
        test_results['Predictions'] = test_results['Predictions']
        test_results['Labels'] = test_results['Labels']
        test_results['Slots'] = slots
        test_results['Ranking'] = [
            ranking
            for ranking in test_results['Ranking']
        ]
        test_results['Prev_actions'] = pre_actions
        df = pd.DataFrame(test_results)
        corrects_ids = []
        ids_checked = {}
        for dialogue_number in dialogue_id:
            if dialogue_number not in ids_checked:
                samples = df[df['Dialogue_ID'] == dialogue_number]['IsCorrect'].tolist()
                ids_checked[dialogue_number] = all(samples)
            corrects_ids.append(ids_checked[dialogue_number])

        test_results['IsCorrectID'] = corrects_ids

    def run(self, data: object = None) -> object:
        assert isinstance(data, pl.Trainer), "Data must be of type Trainer"
        test = data.test(datamodule=self.data)
        test_results = data.model.test_results
        test_results = pd.DataFrame(test_results)
        test = pd.DataFrame(test)
        """self._update_test_results(test_results)

        test_results = pd.DataFrame(test_results)
        test_results = test_results[[
            'Index',
            'Inputs',
            'Embeddings',
            'Dialogue_ID',
            'IsCorrectID',
            'IsCorrect',
            'Intentions',
            'Predictions',
            'Labels',
            'Slots',
            'Prev_actions',
            'Ranking',
        ]]"""
        wandb.finish()
        return test_results, test
