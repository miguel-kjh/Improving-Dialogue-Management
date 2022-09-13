from copy import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.dialogue_state_tracker.StateTracker import StateTracker
from models.dialogue_state_tracker.WindowStack import WindowStack
from service.InputOutput.MongoDB import MongoDB


def map_index_to_one_hot(index: int, r: int = 3, k: int = 4) -> int:
    """
    Map an index to a one-hot vector using arithmetic series formula
    :param index:
    :param r:
    :param k:
    :return:
    """
    return k + (index - 1) * r


def rse_2(
        mandatory_slot,
        mandatory_slots,
        mandatory_slot_value,
        optional_slot,
        optional_slots,
        optional_slot_value,
        slot_stored,
        numerical_factor: int = 3
) -> tuple:

    len_mandatory_slots = len(mandatory_slots) * numerical_factor
    len_optional_slots = len(optional_slots) * numerical_factor

    mandatory_slot_embedding = np.zeros(len_mandatory_slots)
    mandatory_slot_embedding[2::3] = 1
    for slot_, value_ in zip(mandatory_slot, mandatory_slot_value):
        if slot_ in slot_stored.keys():
            if slot_stored[slot_] == value_:
                mandatory_slot_embedding[map_index_to_one_hot(mandatory_slots.index(slot_), k=3)] = 0.0
                mandatory_slot_embedding[map_index_to_one_hot(mandatory_slots.index(slot_))] = 1.0
            else:
                mandatory_slot_embedding[map_index_to_one_hot(mandatory_slots.index(slot_), k=3)] = 1.0
                mandatory_slot_embedding[map_index_to_one_hot(mandatory_slots.index(slot_))] = 0.0
        else:
            mandatory_slot_embedding[map_index_to_one_hot(mandatory_slots.index(slot_), k=3)] = 0.0
            mandatory_slot_embedding[map_index_to_one_hot(mandatory_slots.index(slot_))] = 1.0
        slot_stored[slot_] = value_

    optional_slot_embedding = np.zeros(len_optional_slots)
    for slot_, value_ in zip(optional_slot, optional_slot_value):
        if slot_ in slot_stored.keys():
            if slot_stored[slot_] == value_:
                optional_slot_embedding[map_index_to_one_hot(optional_slots.index(slot_), k=3)] = 0.0
                optional_slot_embedding[map_index_to_one_hot(optional_slots.index(slot_))] = 1.0
            else:
                optional_slot_embedding[map_index_to_one_hot(optional_slots.index(slot_), k=3)] = 1.0
                optional_slot_embedding[map_index_to_one_hot(optional_slots.index(slot_))] = 0.0
        else:
            optional_slot_embedding[map_index_to_one_hot(optional_slots.index(slot_), k=3)] = 0.0
            optional_slot_embedding[map_index_to_one_hot(optional_slots.index(slot_))] = 1.0
        slot_stored[slot_] = value_

    return mandatory_slot_embedding, optional_slot_embedding


def rse_1(
        mandatory_slot,
        mandatory_slots,
        mandatory_slot_value,
        optional_slot,
        optional_slots,
        optional_slot_value,
        slot_stored
) -> tuple:
    len_mandatory_slots = len(mandatory_slots)
    len_optional_slots = len(optional_slots)

    mandatory_slot_embedding = np.zeros(len_mandatory_slots)
    for slot_, value_ in zip(mandatory_slot, mandatory_slot_value):
        mandatory_slot_embedding[mandatory_slots.index(slot_)] = 1.0
        slot_stored[slot_] = value_

    optional_slot_embedding = np.zeros(len_optional_slots)
    for slot_, value_ in zip(optional_slot, optional_slot_value):
        optional_slot_embedding[optional_slots.index(slot_)] = 1.0
        slot_stored[slot_] = value_

    return mandatory_slot_embedding, optional_slot_embedding


class RseStateTracker(StateTracker):
    """
    This class implements the RSE state tracker. The RSE (representative slots embeddings)
    is a codification of the state of the dialogue. The state is represented by a vector formed by
    a concatenation of the following elements:
        - The intention of the user
        - The slots of the user, these slots are separate in two groups: the mandatory slots and the optional slots
        - The previous action of the system
    """

    def __init__(self):
        super().__init__()
        self._mandatory_slot_column = 'Mandatory Slots'
        self._mandatory_slot_column_value = 'Mandatory Slots Value'
        self._optional_slot_column = 'Optional Slots'
        self._optional_slot_column_value = 'Optional Slots Value'

    def _get_embedding(
            self,
            intention: list,
            intentions: list,
            mandatory_slot: list,
            mandatory_slot_value: list,
            mandatory_slots: list,
            optional_slot: list,
            optional_slot_value: list,
            optional_slots: list,
            action: str,
            actions: list,
            slot_stored: dict
    ):

        len_intentions = len(intentions)
        len_actions = len(actions)

        intention_embedding = np.zeros(len_intentions)
        for intent in intention:
            intention_embedding[intentions.index(intent)] = 1

        mandatory_slot_embedding, optional_slot_embedding = rse_2(
            mandatory_slot,
            mandatory_slots,
            mandatory_slot_value,
            optional_slot,
            optional_slots,
            optional_slot_value,
            slot_stored
        )

        action_embedding = np.zeros(len_actions)
        action_embedding[actions.index(action)] = 1

        return np.hstack(
            (
                intention_embedding,
                mandatory_slot_embedding,
                optional_slot_embedding,
                action_embedding
            )
        ).tolist()

    def get_state_and_actions(
            self,
            df_data_oring: pd.DataFrame,
            column_for_intentions,
            column_for_actions,
            mx_history_length=None
    ) -> pd.DataFrame:

        mx_history_length = RseStateTracker._check_mx_history_management(mx_history_length)

        df_data = df_data_oring.copy()
        actions = sorted(list(set(np.hstack(df_data[column_for_actions].values))))
        intents = sorted(list(set(np.hstack(df_data[column_for_intentions].values))))
        mandatory_slots = sorted(list(set(np.hstack(df_data[self._mandatory_slot_column].values))))
        print('Mandatory slots: ', mandatory_slots)
        optional_slots = sorted(list(set(np.hstack(df_data[self._optional_slot_column].values))))
        print('Optional slots: ', optional_slots)
        len_embedding = len(actions) + len(intents) + (len(mandatory_slots) + len(optional_slots)) * 3

        dialogue_state = self._get_schema_dialogue_state_dataset()

        df_data = df_data.groupby(by='Dialogue ID')
        for id_, df_group in tqdm(df_data, desc='RseStateTracker'):
            last_action = 'LISTEN'
            window = WindowStack(mx_history_length, len_embedding)
            store_slots = {}
            for row in df_group.to_dict('records'):
                action = row[column_for_actions][0]
                total_slots = row[self._mandatory_slot_column] + row[self._optional_slot_column]
                window.add(self._get_embedding(
                    row[column_for_intentions],
                    intents,
                    row[self._mandatory_slot_column],
                    row[self._mandatory_slot_column_value],
                    mandatory_slots,
                    row[self._optional_slot_column],
                    row[self._optional_slot_column_value],
                    optional_slots,
                    last_action,
                    actions,
                    store_slots
                ))
                self._add_state_to_schema(
                    dialogue_state,
                    id_,
                    row[column_for_intentions],
                    total_slots,
                    last_action,
                    action,
                    row[column_for_actions],
                    row['Type'],
                    window.get_stack()
                )
                last_action = copy(action)
                for action in row[column_for_actions][1:]:
                    window.add(self._get_embedding(
                        [],
                        intents,
                        row[self._mandatory_slot_column],
                        row[self._mandatory_slot_column_value],
                        mandatory_slots,
                        row[self._optional_slot_column],
                        row[self._optional_slot_column_value],
                        optional_slots,
                        last_action,
                        actions,
                        store_slots
                    ))
                    self._add_state_to_schema(
                        dialogue_state,
                        id_,
                        [],
                        total_slots,
                        last_action,
                        action,
                        row[column_for_actions],
                        row['Type'],
                        window.get_stack()
                    )
                    last_action = copy(action)

        df = pd.DataFrame(dialogue_state)
        df['Index'] = df.index
        return df


def main():
    mongodb_service = MongoDB("synthetic_dialogues", "mongodb://localhost:27017")
    df = mongodb_service.load("syn_example_3_change_idea_ALL")
    assert not df.empty, "Dataframe is empty"
    state_tracker = RseStateTracker()
    column_for_intentions = 'Atomic Intent'
    column_for_actions = 'Action'
    max_history_length = 2
    df = state_tracker.get_state_and_actions(
        df,
        column_for_intentions=column_for_intentions,
        column_for_actions=column_for_actions,
        mx_history_length=max_history_length
    )
    embeddings = df['State'].tolist()
    len_embedding = len(embeddings[0])
    for embedding in embeddings[1:]:
        if len_embedding != len(embedding):
            print(f'Error en el embedding: {len_embedding} != {len(embedding)}')
    print('Todas las listas de embeddings tienen la misma shape')
    embeddings = np.array(df['State'].tolist())
    print(embeddings.shape)
    print(embeddings[0][0])


if __name__ == '__main__':
    main()
