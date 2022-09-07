from copy import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.dialogue_state_tracker.StateTracker import StateTracker
from models.dialogue_state_tracker.WindowStack import WindowStack
from service.InputOutput.MongoDB import MongoDB


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
        self._optional_slot_column = 'Optional Slots'

    @staticmethod
    def _get_embedding(
            intention: list,
            intentions: list,
            mandatory_slot: list,
            mandatory_slots: list,
            optional_slot: list,
            optional_slots: list,
            action: str,
            actions: list,
            len_intentions: int,
            len_mandatory_slots: int,
            len_optional_slots: int,
            len_actions: int
    ):

        intention_embedding = np.zeros(len_intentions)
        for intent in intention:
            intention_embedding[intentions.index(intent)] = 1

        mandatory_slot_embedding = np.ones(len_mandatory_slots*2)
        mandatory_slot_embedding[::2] = 0
        for slot_ in mandatory_slot:
            mandatory_slot_embedding[mandatory_slots.index(slot_)*2] = 1
        optional_slot_embedding = np.zeros(len_optional_slots*2)
        for slot_ in optional_slot:
            optional_slot_embedding[optional_slots.index(slot_)*2] = 1

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
        len_embedding = len(actions) + len(intents) + len(mandatory_slots) + len(optional_slots)

        dialogue_state = self._get_schema_dialogue_state_dataset()

        df_data = df_data.groupby(by='Dialogue ID')
        for id_, df_group in tqdm(df_data, desc='RseStateTracker'):
            last_action = 'LISTEN'
            window = WindowStack(mx_history_length, len_embedding)
            for row in df_group.to_dict('records'):
                action = row[column_for_actions][0]
                total_slots = row[self._mandatory_slot_column] + row[self._optional_slot_column]
                window.add(self._get_embedding(
                    row[column_for_intentions],
                    intents,
                    row[self._mandatory_slot_column],
                    mandatory_slots,
                    row[self._optional_slot_column],
                    optional_slots,
                    last_action,
                    actions,
                    len(intents),
                    len(mandatory_slots),
                    len(optional_slots),
                    len(actions)
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
                        [],
                        row[self._mandatory_slot_column],
                        mandatory_slots,
                        row[self._optional_slot_column],
                        optional_slots,
                        last_action,
                        actions,
                        len(intents),
                        len(mandatory_slots),
                        len(optional_slots),
                        len(actions)
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
        print(df['State'].values[0])
        return df


def main():
    mongodb_service = MongoDB("synthetic_dialogues", "mongodb://localhost:27017")
    df = mongodb_service.load("syn_example_2_optional_slots_chit_chat_ALL")
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
    #encuentra un estado que no tenga la misma shape
    df['State'] = df['State'].apply(lambda x: np.array(x))
    embeddings = np.array(df['State'].tolist())
    print(embeddings.shape)
#    df.to_csv('state_tracker.csv', index=False)

    """mongodb_service.save(df, f"SGD_dataset_TINY_state_tracker_{column_for_intentions}_{column_for_actions}_"
                             f"max_history={max_history_length}")"""


if __name__ == '__main__':
    main()
