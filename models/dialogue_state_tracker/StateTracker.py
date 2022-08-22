from copy import copy

import pandas as pd
import numpy as np
from tqdm import tqdm

from models.dialogue_state_tracker.WindowStack import WindowStack
from service.MongoDB import MongoDB


# TODO: add slots to the embeddings with actions only
class StateTracker:

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

    @staticmethod
    def _get_schema_dialogue_state_dataset() -> dict:
        return {
            'Dialogue_ID': [],
            'Intention': [],
            'Slots': [],
            'Prev_action': [],
            'Label': [],
            'Set_Label': [],
            'Type': [],
            'State': []
        }

    @staticmethod
    def _add_state_to_schema(
            schema: dict,
            id: str,
            intention: list,
            slots: list,
            prev_action: str,
            label: str,
            set_label: str,
            type: str,
            state: np.array
    ):
        schema['Dialogue_ID'].append(id)
        schema['Intention'].append(intention)
        schema['Slots'].append(slots)
        schema['Prev_action'].append(prev_action)
        schema['Label'].append(label)
        schema['Set_Label'].append(set_label)
        schema['Type'].append(type)
        schema['State'].append(state)

    def get_state_and_actions(
            self,
            df_data_oring: pd.DataFrame,
            column_for_intentions,
            column_for_actions,
            mx_history_length=None
    ) -> pd.DataFrame:

        assert mx_history_length > 0, "mx_history_length must be greater than 0"

        if mx_history_length is None:
            mx_history_length = 1

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

        return pd.DataFrame(dialogue_state)


def main():
    mongodb_service = MongoDB("multi_woz", "mongodb://localhost:27017")
    df = mongodb_service.load("multi_woz_dataset_TINY")
    assert not df.empty, "Dataframe is empty"
    state_tracker = StateTracker()
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
    df.to_csv('SGD_dataset_TINY_state_tracker.csv', index=False)

    """mongodb_service.save(df, f"SGD_dataset_TINY_state_tracker_{column_for_intentions}_{column_for_actions}_"
                             f"max_history={max_history_length}")"""


if __name__ == '__main__':
    main()
