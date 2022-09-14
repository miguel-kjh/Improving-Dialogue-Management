import copy
import os
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import List

from service.Pipeline import Pipeline
from service.SgdDataModule import SgdDataModule

from models.dialogue_state_tracker.BinaryStateTracker import BinaryStateTracker
from models.dialogue_state_tracker.RseStateTracker import RseStateTracker
from models.dialogue_policy.supervised_learning.TedPolicy import Ted
from models.dialogue_policy.supervised_learning.LedPolicy import Led
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from service.InputOutput.MongoDB import MongoDB
from models.dialogue_policy.supervised_learning.StarSpacePolicy import StarSpacePolicy
from view.Logger import Logger


class TrainAndEvaluateService(Pipeline):

    def __init__(self, configuration: dict):
        super().__init__()
        self.configuration = copy.deepcopy(configuration)
        self.mongodb_service = MongoDB(configuration['dataset']['DB_name'], configuration['database'][0]['path'])
        self.state_tracker = None
        embedding_type = self.configuration['model']['embedding_type']
        class_correction = self.configuration['model']['class_correction']
        if embedding_type == 'binary':
            self.state_tracker = BinaryStateTracker(class_correction)
        elif embedding_type == 'rse':
            self.state_tracker = RseStateTracker(class_correction)
        else:
            raise ValueError(f'Unknown embedding type: {embedding_type}')

        self.name = self.configuration['dataset']['name']
        domain = self.configuration['dataset']['domain']
        column_for_intentions = self.configuration['dataset']['intention']
        column_for_actions = self.configuration['dataset']['action']
        max_history_length = self.configuration['dataset']['max_history']
        dataset_name = f"{self.name}_{domain}"
        model = self.configuration['model']['name']
        self.name_experiment = f"{model}_{self.name}_{domain}_{column_for_intentions}_{column_for_actions}" \
                               f"_{max_history_length}_{embedding_type}"
        #self.dataset = self.mongodb_service.load(file_dataset)
        Logger.info("Create new dataset with that configuration")
        df = self.mongodb_service.load(dataset_name)
        assert not df.empty, f"Dataset {os.path.join(configuration['database'][0]['path'], configuration['dataset']['name'], dataset_name)} is empty"

        self.dataset = self.state_tracker.get_state_and_actions(
            df,
            column_for_intentions=column_for_intentions,
            column_for_actions=column_for_actions,
            mx_history_length=max_history_length
        )
        self.embeddings = np.array(self.dataset['State'].tolist())
        self.labels = self.dataset['Label'].tolist()
        self.action_encoder = LabelEncoder()
        self.actions = self.action_encoder.fit_transform(self.labels)
        self.class_actions = sorted(list(set(self.actions)))
        self.actions_one_hot_encoder = OneHotEncoder(sparse=False)
        self.actions_one_hot_encoder.fit(self.actions.reshape(-1, 1))
        self.num_classes = len(self.action_encoder.classes_)
        self.activate_wandb_logging = self.configuration['resources']['wandb']
        self.path_results = os.path.join(
            "results",
            self.name_experiment
        )
        self._create_folder(self.path_results)

    def get_path_results(self) -> str:
        return self.path_results

    @staticmethod
    def _create_folder(path: str):

        if os.path.exists(path):
            shutil.rmtree(path)

        Logger.info("Creating folder: " + path)
        os.makedirs(path)

    @staticmethod
    def get_model(model: str, config: dict, actions: List[int]) -> pl.LightningModule:
        models = {
            "TED": Ted,
            "LED": Led,
            "SS": StarSpacePolicy
        }
        return models[model](config, actions)

    @staticmethod
    def __get_callbacks() -> list:
        callbacks = [
            EarlyStopping(
                monitor='val_f1',
                verbose=True,
                mode='max'
            )
        ]

        return callbacks

    def _fit(self, model: pl.LightningModule, data: SgdDataModule) -> pl.Trainer:
        wandb_logger = WandbLogger(project=self.name, name=self.name_experiment) \
            if self.activate_wandb_logging else None

        trainer = pl.Trainer(
            gpus=self.configuration['resources']['gpus'],
            logger=wandb_logger,
            max_epochs=self.configuration['model']['epochs'],
            #callbacks=self.__get_callbacks()
        )

        trainer.fit(model, data)

        return trainer

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
        test_results['Predictions'] = self.action_encoder.inverse_transform(test_results['Predictions'])
        test_results['Labels'] = self.action_encoder.inverse_transform(test_results['Labels'])
        test_results['Ranking'] = [
            self.action_encoder.inverse_transform(ranking)
            for ranking in test_results['Ranking']
        ]
        intentions, slots, pre_actions, dialogue_id = self._transform_binary_embeddings(test_results['Index'])
        test_results['Intentions'] = intentions
        test_results['Slots'] = slots
        test_results['Prev_actions'] = pre_actions
        test_results['Dialogue_ID'] = dialogue_id

    def _update_actions_results(self, actions_results: dict):
        actions_results['Actions'] = self.action_encoder.inverse_transform(actions_results['Actions'])

    def _evaluate(self, trainer: pl.Trainer, data: SgdDataModule) -> None:
        metrics_results = trainer.test(datamodule=data)
        test_results = trainer.model.test_results
        self._update_test_results(test_results)

        test_results = pd.DataFrame(test_results)
        metrics_results = pd.DataFrame(metrics_results)
        Logger.info('Save results')
        test_results_file = os.path.join(
            self.path_results,
            'embeddings.csv'
        )
        metrics_results_file = os.path.join(
            self.path_results,
            'metrics.csv'
        )
        self.output_csv_service.save(
            test_results,
            test_results_file
        )
        self.output_csv_service.save(
            metrics_results,
            metrics_results_file
        )

        actions_results = trainer.model.actions_results
        self._update_actions_results(actions_results)

        actions_results = pd.DataFrame(actions_results)
        actions_results_file = os.path.join(
            self.path_results,
            'actions.csv'
        )
        self.output_csv_service.save(
            actions_results,
            actions_results_file
        )

        if self.activate_wandb_logging:
            wandb.finish()

    def _get_samples_by_dataset(self, type_data: str) -> tuple:
        dataset = self.dataset[self.dataset['Type'] == type_data]
        indexes = dataset['Index'].to_list()
        labels = self.action_encoder.transform(dataset['Label'])
        set_labels = [
            list(set(self.action_encoder.transform(actions)))
            for actions in dataset['Set_Label']
        ]
        max_len_of_set_labels = max([len(set_of_label) for set_of_label in set_labels])
        set_labels = [
            set_of_label + [-1] * (max_len_of_set_labels - len(set_of_label))
            for set_of_label in set_labels
        ]
        features = np.array(dataset['State'].to_list())

        assert len(indexes) == len(features) == len(labels) == len(set_labels), \
            f"Bab dimension of features: {len(features)} - " \
            f"labels : {len(labels)} - set_labels: {len(set_labels)} " \
            f"index: {len(indexes)}"
        numbers = [len(set_label) for set_label in set_labels]
        assert max(numbers) == min(numbers), "Bab dimension of set_labels"

        return features, labels, set_labels, indexes

    def process(self):
        n_features = self.embeddings[0].shape[1]
        self.configuration['model']['n_features'] = n_features
        self.configuration['model']['hidden_layers_sizes_pre_dial'][0][0] = n_features

        model = self.get_model(
            self.configuration['model']['name'],
            self.configuration['model'],
            self.class_actions
        )

        train, train_labels, set_train_labels, train_indexes = self._get_samples_by_dataset('train')
        validation, validation_labels, set_validation_labels, validation_indexes = self._get_samples_by_dataset('dev')
        test, test_labels, set_test_labels, test_indexes = self._get_samples_by_dataset('test')

        sgd_module = SgdDataModule(
            train,
            train_labels,
            set_train_labels,
            validation,
            validation_labels,
            set_validation_labels,
            test,
            test_labels,
            set_test_labels,
            train_indexes,
            validation_indexes,
            test_indexes,
            batch_size=self.configuration['model']['batch_size']
        )

        trainer = self._fit(model, sgd_module)
        self._evaluate(trainer, sgd_module)
