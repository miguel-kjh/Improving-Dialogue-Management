import pandas as pd
import wandb
from typing import List

from tqdm import tqdm

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

    def _update_test_results(self, test_results: dict):
        intentions, entities, slots, pre_actions, dialogue_id = self._transform_binary_embeddings(test_results['Index'])
        test_results['Dialogue_ID'] = dialogue_id
        test_results['Intentions'] = intentions
        test_results['Entities'] = entities
        test_results['Pre_actions'] = pre_actions
        test_results['Slots'] = slots

    def run(self, data: object = None) -> object:
        assert isinstance(data, pl.Trainer), "Data must be of type Trainer"
        test = data.test(datamodule=self.data)
        test_results = data.model.test_results
        self._update_test_results(test_results)
        test = pd.DataFrame(test)
        test_results = pd.DataFrame(test_results)
        wandb.finish()
        return test_results, test
