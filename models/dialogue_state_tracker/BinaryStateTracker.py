from copy import copy

import pandas as pd
import numpy as np
from tqdm import tqdm

from models.dialogue_state_tracker.StateTracker import StateTracker
from models.dialogue_state_tracker.WindowStack import WindowStack
from service.InputOutput.MongoDB import MongoDB


class BinaryStateTracker(StateTracker):

    def __init__(self, class_correction: bool = False):
        super().__init__(class_correction)

    @staticmethod
    def _get_embedding(
            intention: list,
            intentions: list,
            mandatory_slot: list,
            mandatory_slots: list,
            mandatory_slots_by_domain: list,
            optional_slot: list,
            optional_slots: list,
            action: str,
            actions: list,
    ):

        len_intentions = len(intentions)
        len_actions = len(actions)
        len_mandatory_slots = len(mandatory_slots)
        len_optional_slots = len(optional_slots)


        intention_embedding = np.zeros(len_intentions)
        for intent in intention:
            intention_embedding[intentions.index(intent)] = 1
        mandatory_slot_embedding = np.zeros(len_mandatory_slots)
        for slot_ in mandatory_slot:
            mandatory_slot_embedding[mandatory_slots.index(slot_)] = 1
        optional_slot_embedding = np.zeros(len_optional_slots)
        for slot_ in optional_slot:
            optional_slot_embedding[optional_slots.index(slot_)] = 1
        action_embedding = np.zeros(len_actions)
        action_embedding[actions.index(action)] = 1

        mandatory_slots_by_domain_idx = [mandatory_slots.index(slot) for slot in mandatory_slots_by_domain]
        is_mandatory_slot_complete = True
        for idx in mandatory_slots_by_domain_idx:
            if mandatory_slot_embedding[idx] == 0:
                is_mandatory_slot_complete = False
                break

        return np.hstack(
            (
                intention_embedding,
                mandatory_slot_embedding,
                optional_slot_embedding,
                action_embedding
            )
        ).tolist(), is_mandatory_slot_complete

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
        mandatory_slots = sorted(list(set(np.hstack(df_data[self._mandatory_slot_column].values))))
        optional_slots = sorted(list(set(np.hstack(df_data[self._optional_slot_column].values))))
        len_embedding = len(actions) + len(intents) + len(mandatory_slots) + len(optional_slots)

        dialogue_state = self._get_schema_dialogue_state_dataset()
        slots_by_domains = self._get_slots_by_domain(df_data)

        df_data = df_data.groupby(by='Dialogue ID')
        for id_, df_group in tqdm(df_data, desc='StateTracker'):
            last_action = 'LISTEN'
            window = WindowStack(mx_history_length, len_embedding)
            for row in df_group.to_dict('records'):
                action = row[column_for_actions][0]
                total_slots = row[self._mandatory_slot_column] + row[self._optional_slot_column]
                emb, is_mandatory_slots = self._get_embedding(
                    row[column_for_intentions],
                    intents,
                    row[self._mandatory_slot_column],
                    mandatory_slots,
                    slots_by_domains[row['Domain']]['mandatory'],
                    row[self._optional_slot_column],
                    optional_slots,
                    last_action,
                    actions
                )
                window.add(emb)
                action = self._change_fuzzy_action(action, is_mandatory_slots)
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
                    emb, is_mandatory_slots = self._get_embedding(
                        [],
                        intents,
                        row[self._mandatory_slot_column],
                        mandatory_slots,
                        slots_by_domains[row['Domain']]['mandatory'],
                        row[self._optional_slot_column],
                        optional_slots,
                        last_action,
                        actions,
                    )
                    window.add(emb)
                    action = self._change_fuzzy_action(action, is_mandatory_slots)
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
    df = mongodb_service.load("syn_example_4_multidomain_ALL")
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
    # df.to_csv('SGD_dataset_TINY_state_tracker.csv', index=False)

    """mongodb_service.save(df, f"SGD_dataset_TINY_state_tracker_{column_for_intentions}_{column_for_actions}_"
                             f"max_history={max_history_length}")"""
    embeddings = np.array(df['State'].tolist())
    print(embeddings.shape)
    print(embeddings[0][0])
    print(embeddings)


if __name__ == '__main__':
    main()
