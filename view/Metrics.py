from abc import abstractmethod

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
from sklearn.manifold import TSNE
from typing import Tuple
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def _plot_figure(data: pd.DataFrame, actions_dict: dict) -> Tuple[Figure, Figure]:
    def add_actions_points(fig: Figure, actions_dict: dict) -> None:
        for act, embeddings in actions_dict.items():
            fig.add_scatter(
                x=[embeddings[0]],
                y=[embeddings[1]],
                mode='markers',
                marker=dict(size=10, color='black'),
                showlegend=False,
                hovertext=[act]
            )

    # plot 2D
    fig1 = px.scatter(
        data, x='x', y='y',
        hover_data=["Is_Correct", "Intentions", "Slots", "Prev_Actions", "Real_labels", "Index"],
        color="Predicted", labels={'color': 'Actions'}
    )

    add_actions_points(fig1, actions_dict)

    fig2 = px.scatter(
        data, x='x', y='y',
        color="Is_Correct", labels={'color': 'Actions'}
    )

    add_actions_points(fig2, actions_dict)

    return fig1, fig2


def _error_management(
        n_components: int,
        perplexity: int,
        features: list,
        predictions: list,
        real_labels: list,
        is_correct_labels: list,
        intentions: list,
        slots: list,
        prev_actions: list,
):
    assert n_components in [2, 3], "n_components must be 2 or 3"
    assert perplexity > 0, "perplexity must be greater than 0"
    assert len(features) == len(predictions), "features and predictions must be the same length"
    assert len(features) == len(real_labels), "features and real_labels must be the same length"
    assert len(features) == len(is_correct_labels), "features and is_correct_labels must be the same length"
    assert len(features) == len(intentions), "features and intentions must be the same length"
    assert len(features) == len(slots), "features and slots must be the same length"
    assert len(features) == len(prev_actions), "features and prev_actions must be the same length"


class Metrics:

    @abstractmethod
    def plot_tsne(
            features: list,
            predictions: list,
            real_labels: list,
            is_correct_labels: list,
            intentions: list,
            slots: list,
            prev_actions: list,
            index: list,
            actions: pd.DataFrame,
            n_components: int = 2,
            perplexity: int = 35,
            random_state=42
    ) -> Tuple[Figure, Figure]:

        _error_management(
            n_components=n_components,
            perplexity=perplexity,
            features=features,
            predictions=predictions,
            real_labels=real_labels,
            is_correct_labels=is_correct_labels,
            intentions=intentions,
            slots=slots,
            prev_actions=prev_actions,
        )

        model = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)

        actions_embeddings = [e for e in actions['Embeddings']]
        latent_embeddings_N_dimensional = features + actions_embeddings
        latent_embeddings_2D = model.fit_transform(latent_embeddings_N_dimensional)

        projections = latent_embeddings_2D[:len(features)]
        actions_embeddings_2D = latent_embeddings_2D[len(features):]

        df = {'x': projections[:, 0].tolist(), 'y': projections[:, 1].tolist(), 'Predicted': predictions,
              'Is_Correct': is_correct_labels, 'Intentions': intentions, 'Slots': slots, 'Prev_Actions': prev_actions,
              'Real_labels': real_labels, 'Index': index}

        # actions to dicts

        actions = actions.to_dict(orient='records')
        actions_dict = {action['Actions']: actions_embeddings_2D[idx] for idx, action in enumerate(actions)}

        return _plot_figure(pd.DataFrame(df), actions_dict)

    @abstractmethod
    def plot_confusion_matrix(
            predictions: list,
            real_labels: list
    ) -> Figure:
        cm = confusion_matrix(real_labels, predictions)
        system_actions = list(set(predictions + real_labels))
        system_actions.sort()
        fig = px.imshow(
            cm,
            labels=dict(x="Label", y="Predict", color="Hits"),
            x=system_actions,
            y=system_actions,
            text_auto=True,
            aspect="auto"
        )

        return fig
