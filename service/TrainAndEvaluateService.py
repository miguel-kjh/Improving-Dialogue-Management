import ast
import os
import shutil

import numpy as np
from numpy import isin
import pytorch_lightning as pl
import pandas as pd
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from sqlalchemy import desc
from tqdm import tqdm
import wandb
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from service.Pipeline.Pipeline import Pipeline
from service.Pipeline.SgdDataModule import SgdDataModule
from utils.Domain import Domain
from utils.ProjectConstants import PROJECT_NAME, NUM_GPUS, SGD_DATASET_RES, METRIC_FOLDER
from sklearn.preprocessing import LabelEncoder
from utils.TrainUtils import get_model, save_plot
from view.Logger import Logger

import seaborn as sn


class TrainAndEvaluateService(Pipeline):

    def __init__(
            self,
            model_config: str,
            domain: Domain,
            filename: str,
            name_experiment: str,
            activate_wandb_logging: bool = False
    ):
        super().__init__()
        self.configuration = self.input_json_service.load(model_config)
        self.file_config = \
            f"{os.path.basename(model_config).replace('.json', '')}_" \
            f"{os.path.basename(filename).replace('.csv', '')}_{domain.name}"
        self.type_feature = self.configuration['config']['type_feature']
        path = filename + "_" + str(domain)
        self.domain = domain
        self.dataset = self.mongodb_service.load(path)
        self.embeddings = np.array(self.dataset[self.type_feature].tolist())
        self.labels = self.dataset['Label'].tolist()
        self.action_encoder = LabelEncoder()
        self.actions = self.action_encoder.fit_transform(self.labels)
        self.num_classes = len(self.action_encoder.classes_)
        self.activate_wandb_logging = activate_wandb_logging
        self.priority = 4
        self.path_results = os.path.join(
            METRIC_FOLDER,
            self.file_config,
        )
        self.name_experiment = name_experiment

        self._create_folder(self.path_results)

    def _create_folder(self, path: str):

        if os.path.exists(path):
            shutil.rmtree(path)

        Logger.info("Creating folder: " + path)
        os.makedirs(path)

    def _fit(self, model: pl.LightningModule, data: SgdDataModule) -> pl.Trainer:
        wandb_logger = WandbLogger(project=PROJECT_NAME, name=self.name_experiment) \
            if self.activate_wandb_logging else None

        trainer = pl.Trainer(
            gpus=NUM_GPUS,
            logger=wandb_logger,
            max_epochs=self.configuration['config']['epochs'],
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
                record_index = self.dataset[self.type_feature].to_list().index(embedding)
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
        features = np.array(dataset[self.type_feature].to_list())


        assert len(features) == len(labels) == len(set_labels), \
            f"Bab dimension of features: {len(features)} - " \
            f"labels : {len(labels)} - set_labels: {len(set_labels)} "
        numbers = [len(set_label) for set_label in set_labels]
        assert max(numbers) == min(numbers), "Bab dimension of set_labels"

        return features, labels, set_labels


    def process(self):
        Logger.print_dict(self.configuration)
        Logger.print_sep()

        features = self.embeddings[0].shape[0]
        self.configuration['config']['n_features'] = features
        self.configuration['config']['hidden_layers_sizes_pre_dial'][0][0] = features

        model = get_model(
            self.configuration['model'],
            self.configuration['config'],
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
            batch_size=self.configuration['config']['batch_size'],
            window_size=self.configuration['config']['window_size']
        )

        trainer = self._fit(model, sgd_module)
        self._evaluate(trainer, sgd_module)
