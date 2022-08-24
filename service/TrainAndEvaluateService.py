import copy
import os
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import wandb

from sklearn.preprocessing import LabelEncoder

from service.OutputCsvService import OutputCsvService
from service.SgdDataModule import SgdDataModule

from models.dialogue_state_tracker.StateTracker import StateTracker
from models.dialogue_policy.supervised_learning.TedPolicy import Ted
from models.dialogue_policy.supervised_learning.LedPolicy import Led
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from service.MongoDB import MongoDB
from view.Logger import Logger


class TrainAndEvaluateService:

    def __init__(self, configuration: dict, create_states: bool = True):
        super().__init__()
        self.configuration = copy.deepcopy(configuration)
        self.mongodb_service = MongoDB(configuration['dataset']['DB_name'], configuration['database'][0]['path'])
        self.state_tracker = StateTracker()

        self.name = self.configuration['dataset']['name']
        domain = self.configuration['dataset']['domain']
        column_for_intentions = self.configuration['dataset']['intention']
        column_for_actions = self.configuration['dataset']['action']
        max_history_length = self.configuration['dataset']['max_history']
        dataset_name = f"{self.name}_{domain}"
        self.name_experiment = f"{self.name}_{domain}_{column_for_intentions}_{column_for_actions}" \
                               f"_{max_history_length}"
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
        #self.mongodb_service.save(self.dataset, file_dataset)
        self.embeddings = np.array(self.dataset['State'].tolist())
        self.labels = self.dataset['Label'].tolist()
        self.action_encoder = LabelEncoder()
        self.actions = self.action_encoder.fit_transform(self.labels)
        self.num_classes = len(self.action_encoder.classes_)
        self.activate_wandb_logging = self.configuration['resources']['wandb']
        self.path_results = os.path.join(
            "results",
            self.name_experiment
        )
        self.output_csv_service = OutputCsvService()

    @staticmethod
    def _create_folder(path: str):

        if os.path.exists(path):
            shutil.rmtree(path)

        Logger.info("Creating folder: " + path)
        os.makedirs(path)

    @staticmethod
    def get_model(model: str, config: dict, num_actions: int) -> pl.LightningModule:
        models = {
            "TED": Ted,
            "LED": Led,
        }

        return models[model](config, num_actions)

    @staticmethod
    def __get_callbacks() -> list:
        callbacks = [
            EarlyStopping(
                monitor='val_acc',
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

    def _transform_binary_embeddings(self, embeddings: list) -> tuple:
        intentions = []
        slots = []
        pre_actions = []

        embeddings_transformed = {}

        for embedding in tqdm(embeddings, desc='Transforming embeddings'):
            embedding2key = str(embedding)
            if embedding2key not in embeddings_transformed.keys():
                print(embedding)
                record_index = self.dataset['State'].to_list().index(embedding)
                embeddings_transformed[embedding2key] = record_index
            else:
                record_index = embeddings_transformed[embedding2key]

            intentions.append(self.dataset['Intention'][record_index])
            slots.append(self.dataset['Slot'][record_index])
            pre_actions.append(self.dataset['Prev_actions'][record_index])

        return intentions, slots, pre_actions

    def _update_test_results(self, test_results: dict):
        test_results['Index'] = list(range(0, len(test_results['Predictions'])))
        test_results['Predictions'] = self.action_encoder.inverse_transform(test_results['Predictions'])
        test_results['Labels'] = self.action_encoder.inverse_transform(test_results['Labels'])
        test_results['Ranking'] = [
            self.action_encoder.inverse_transform(ranking)
            for ranking in test_results['Ranking']
        ]
        intentions, slots, pre_actions = self._transform_binary_embeddings(test_results['Inputs'])
        test_results['Intentions'] = intentions
        test_results['Slots'] = slots
        test_results['Prev_actions'] = pre_actions

    def _update_actions_results(self, actions_results: dict):
        actions_results['Actions'] = self.action_encoder.inverse_transform(actions_results['Actions'])

    def _evaluate(self, trainer: pl.Trainer, data: SgdDataModule) -> None:
        trainer.test(datamodule=data)
        test_results = trainer.model.test_results
        self._update_test_results(test_results)

        test_results = pd.DataFrame(test_results)
        Logger.info('Save results')
        test_results_file = os.path.join(
            self.path_results,
            'embeddings.csv'
        )
        self.output_csv_service.save(
            test_results,
            test_results_file
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

        assert len(features) == len(labels) == len(set_labels), \
            f"Bab dimension of features: {len(features)} - " \
            f"labels : {len(labels)} - set_labels: {len(set_labels)} "
        numbers = [len(set_label) for set_label in set_labels]
        assert max(numbers) == min(numbers), "Bab dimension of set_labels"

        return features, labels, set_labels

    def process(self):
        n_features = self.embeddings[0].shape[1]
        self.configuration['model']['n_features'] = n_features
        self.configuration['model']['hidden_layers_sizes_pre_dial'][0][0] = n_features

        model = self.get_model(
            self.configuration['model']['name'],
            self.configuration['model'],
            self.num_classes
        )

        train, train_labels, set_train_labels = self._get_samples_by_dataset('train')
        validation, validation_labels, set_validation_labels = self._get_samples_by_dataset('dev')
        test, test_labels, set_test_labels = self._get_samples_by_dataset('test')

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
            batch_size=self.configuration['model']['batch_size']
        )

        trainer = self._fit(model, sgd_module)
        self._evaluate(trainer, sgd_module)
