import ast
import os

import pandas as pd

from service.Pipeline import Pipeline
from view.Logger import Logger
from view.Metrics import Metrics


def _transform_dataframe(pd_df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for column in columns:
        try:
            pd_df[column] = pd_df[column].apply(ast.literal_eval)
        except SyntaxError:
            try:
                pd_df[column] = pd_df[column].apply(lambda x: ast.literal_eval(x.replace(" ", ',')))
            except SyntaxError:
                pd_df[column] = pd_df[column].apply(lambda x: ast.literal_eval(x.replace(" ", ',')))
    return pd_df


class MetricService(Pipeline):

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.embeddings_file = os.path.join(self.path, 'embeddings.csv')
        self.actions_file = os.path.join(self.path, 'actions.csv')
        self.columns_for_embeddings = ['Inputs', 'Embeddings', 'Ranking', 'Intentions', 'Slots']
        self.columns_for_actions = ['Inputs', 'Embeddings']
        self.latent_space_name = 'latent_space'

    def run(self):
        Logger.info("Read file: " + self.path)
        pd_df = self.input_csv_service.load(self.embeddings_file)
        pd_df_actions = self.input_csv_service.load(self.actions_file)
        pd_df = _transform_dataframe(pd_df.copy(), self.columns_for_embeddings)
        pd_df_actions = _transform_dataframe(pd_df_actions.copy(), self.columns_for_actions)
        Logger.info("Processing metrics")
        f1, f2 = Metrics.plot_tsne(
            pd_df['Embeddings'].to_list(),
            pd_df['Predictions'].to_list(),
            pd_df['Labels'].to_list(),
            pd_df['IsCorrect'].to_list(),
            pd_df['Intentions'].to_list(),
            pd_df['Slots'].to_list(),
            pd_df['Prev_actions'].to_list(),
            pd_df['Index'].to_list(),
            pd_df_actions
        )

        filename = self.embeddings_file.replace('.csv', '_%s_%s.html' % ('TSNE', self.latent_space_name))
        self.output_html_service.save(f1, filename)

        Logger.info("Confusion matrix")
        cm = Metrics.plot_confusion_matrix(
            pd_df['Labels'].to_list(),
            pd_df['Predictions'].to_list(),
            title=os.path.basename(self.path)
        )
        filename = self.embeddings_file.replace('.csv', '_confusion_matrix.jpg')
        self.output_jpg_service.save(cm, filename)


