from copy import copy

import pandas as pd
import numpy as np
from tqdm import tqdm

from service.MongoDB import MongoDB


class WindowStack:

    def __init__(self, window_size: int, features_dim: int):
        self.window_size = window_size
        self.stack = [np.zeros(features_dim) for _ in range(window_size)]

    def add(self, element: np.array) -> None:
        if len(self.stack) >= self.window_size:
            self.stack.pop(0)
        self.stack.append(element)

    def __len__(self):
        return len(self.stack)

    def get_stack(self) -> np.array:
        return self.stack  # list(reversed(self.stack))


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

        return np.hstack((intention_embedding, slot_embedding, action_embedding)).tolist()

    @staticmethod
    def _get_schema_dialogue_state_dataset() -> dict:
        return {
            'Dialogue_ID': [],
            'Intention': [],
            'Slots': [],
            'Prev_action': [],
            'Label': [],
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
            type: str,
            state: np.array
    ):
        schema['Dialogue_ID'].append(id)
        schema['Intention'].append(intention)
        schema['Slots'].append(slots)
        schema['Prev_action'].append(prev_action)
        schema['Label'].append(label)
        schema['Type'].append(type)
        schema['State'].append(state)

    def get_state_and_actions(
            self,
            df_data_oring: pd.DataFrame,
            column_for_intentions,
            column_for_actions,
            mx_history_length=None
    ) -> pd.DataFrame:

        assert mx_history_length > 1, "mx_history_length must be greater than 1"

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
                    row['Intention'],
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
                    row['Intention'],
                    row['Slots'],
                    last_action,
                    action,
                    row['Type'],
                    window.get_stack()
                )
                last_action = copy(action)
                for action in row[column_for_actions]:
                    window.add(self._get_embedding(
                        [],
                        intents,
                        [],
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
                        [],
                        last_action,
                        action,
                        row['Type'],
                        window.get_stack()
                    )
                    last_action = copy(action)

        return pd.DataFrame(dialogue_state)


def main():
    mongodb_service = MongoDB()
    df = mongodb_service.load("SGD_dataset_TINY")
    state_tracker = StateTracker()
    column_for_intentions = 'Intention'
    column_for_actions = 'Action'
    max_history_length = 5
    """df = state_tracker.get_state_and_actions(
        df,
        column_for_intentions=column_for_intentions,
        column_for_actions=column_for_actions,
        mx_history_length=4
    )

    mongodb_service.save(df, f"SGD_dataset_TINY_state_tracker_{column_for_intentions}_{column_for_actions}_"
                             f"max_history={max_history_length}")"""

    df = mongodb_service.load(f"SGD_dataset_TINY_state_tracker_{column_for_intentions}_{column_for_actions}_"
                              f"max_history={10}")
    print(df.head())


if __name__ == '__main__':
    main()
