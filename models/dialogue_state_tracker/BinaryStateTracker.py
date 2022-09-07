from copy import copy

import pandas as pd
import numpy as np
from tqdm import tqdm

from models.dialogue_state_tracker.StateTracker import StateTracker
from models.dialogue_state_tracker.WindowStack import WindowStack
from service.InputOutput.MongoDB import MongoDB


# TODO: add slots to the embeddings with actions only
class BinaryStateTracker(StateTracker):

    @staticmethod
    def _get_embedding(
            intention: list,
            intentions: list,
            slot: list,
            slots: list,
            action: str,
            actions: list,
            len_intentions: int,
            len_slots: int,
            len_actions: int
    ):

        intention_embedding = np.zeros(len_intentions)
        for intent in intention:
            intention_embedding[intentions.index(intent)] = 1
        slot_embedding = np.zeros(len_slots)
        for slot_ in slot:
            slot_embedding[slots.index(slot_)] = 1
        action_embedding = np.zeros(len_actions)
        action_embedding[actions.index(action)] = 1

        return np.hstack(
            (
                intention_embedding,
                slot_embedding,
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

        mx_history_length = BinaryStateTracker._check_mx_history_management(mx_history_length)

        df_data = df_data_oring.copy()
        actions = sorted(list(set(np.hstack(df_data[column_for_actions].values))))
        intents = sorted(list(set(np.hstack(df_data[column_for_intentions].values))))
        slots = sorted(list(set(np.hstack(df_data['Slots'].values))))
        len_embedding = len(actions) + len(intents) + len(slots)

        dialogue_state = self._get_schema_dialogue_state_dataset()

        df_data = df_data.groupby(by='Dialogue ID')
        for id_, df_group in tqdm(df_data, desc='StateTracker'):
            last_action = 'LISTEN'
            window = WindowStack(mx_history_length, len_embedding)
            for row in df_group.to_dict('records'):
                action = row[column_for_actions][0]
                window.add(self._get_embedding(
                    row[column_for_intentions],
                    intents,
                    row['Slots'],
                    slots,
                    last_action,
                    actions,
                    len(intents),
                    len(slots),
                    len(actions)
                ))
                self._add_state_to_schema(
                    dialogue_state,
                    id_,
                    row[column_for_intentions],
                    row['Slots'],
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
                        row['Slots'],
                        slots,
                        last_action,
                        actions,
                        len(intents),
                        len(slots),
                        len(actions)
                    ))
                    self._add_state_to_schema(
                        dialogue_state,
                        id_,
                        [],
                        row['Slots'],
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
    mongodb_service = MongoDB("SGD", "mongodb://localhost:27017")
    df = mongodb_service.load("SGD_dataset_TINY")
    print(set(df['Type']))
    assert not df.empty, "Dataframe is empty"
    state_tracker = BinaryStateTracker()
    column_for_intentions = 'Atomic Intent'
    column_for_actions = 'Action'
    max_history_length = 5
    df = state_tracker.get_state_and_actions(
        df,
        column_for_intentions=column_for_intentions,
        column_for_actions=column_for_actions,
        mx_history_length=max_history_length
    )
    print(len(df['State'][0]), len(df['State'][0][0]))
    #df.to_csv('SGD_dataset_TINY_state_tracker.csv', index=False)

    """mongodb_service.save(df, f"SGD_dataset_TINY_state_tracker_{column_for_intentions}_{column_for_actions}_"
                             f"max_history={max_history_length}")"""


if __name__ == '__main__':
    main()
