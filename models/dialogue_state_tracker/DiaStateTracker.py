from copy import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.dialogue_state_tracker.StateTracker import StateTracker
from service.InputOutput.MongoDB import MongoDB


class DiaStateTracker(StateTracker):

    def __init__(self, class_correction: bool = False):
        super().__init__(class_correction)

    def _get_embedding(
            self,
            entities: list,
            db_entities: list,
            intentions: list,
            db_intentions: list,
            prev_action: list,
            db_prev_action: list,
            tasks: list,
            db_tasks: list,
            slots: list,
            db_slots: list,
    ) -> np.array:

        emb = np.zeros(len(db_entities) + len(db_intentions) + len(db_prev_action) + len(db_tasks) + len(db_slots))
        for entity in entities:
            emb[db_entities.index(entity)] = 1
        for intention in intentions:
            emb[len(db_entities) + db_intentions.index(intention)] = 1
        for task in tasks:
            emb[len(db_entities) + len(db_intentions) + db_tasks.index(task)] = 1
        for slot in slots:
            emb[len(db_entities) + len(db_intentions) + len(db_tasks) + db_slots.index(slot)] = 1
        if prev_action is not None:
            for action in prev_action:
                emb[len(db_entities) + len(db_intentions) + len(db_tasks) + len(db_slots) + db_prev_action.index(
                    action)] = 1
        return emb

    def get_state_and_actions(
            self,
            df_data_oring: pd.DataFrame,
            column_for_intentions,
            column_for_actions,
            mx_history_length=None
    ) -> pd.DataFrame:

        df_data = df_data_oring.copy()
        actions_db = sorted(list(set(np.hstack(df_data[column_for_actions].values))))
        intents = sorted(list(set(np.hstack(df_data[column_for_intentions].values))))
        slots = sorted(list(set(np.hstack(df_data[self._slots_column].values))))
        mandatory_slots = sorted(list(set(np.hstack(df_data[self._mandatory_slot_column].values))))
        optional_slots = sorted(list(set(np.hstack(df_data[self._optional_slot_column].values))))
        entities = sorted(list(set(np.hstack(df_data[self._entity_column].values))))
        tasks = sorted(list(set(np.hstack(df_data[self._task_column].values))))
        dialogue_state = self._get_schema_dialogue_state_dataset()
        slots_by_domains = self._get_slots_by_domain(df_data)

        df_data = df_data.groupby(by='Dialogue ID')
        for id_, df_group in tqdm(df_data, desc='StateTracker'):
            last_action = None
            for row in df_group.to_dict('records'):
                total_slots = row[self._slots_column]
                actions = row[column_for_actions]
                emb = self._get_embedding(
                    entities=row['Entities'],
                    db_entities=entities,
                    intentions=row[column_for_intentions],
                    db_intentions=intents,
                    prev_action=last_action,
                    db_prev_action=actions_db,
                    tasks=[row['Task']],
                    db_tasks=tasks,
                    slots=total_slots,
                    db_slots=slots
                )
                y_actions = np.zeros(len(actions_db))
                for idx, action in enumerate(row[column_for_actions]):
                    y_actions[idx] = actions_db.index(action) + 1
                self._add_state_to_schema(
                    dialogue_state,
                    id_,
                    row[column_for_intentions],
                    total_slots,
                    last_action,
                    actions,
                    y_actions,
                    row['Type'],
                    emb
                )
                last_action = copy(actions)

        df = pd.DataFrame(dialogue_state)
        df['Index'] = df.index
        return df


def main():
    mongodb_service = MongoDB("SGD", "mongodb://localhost:27017")
    df = mongodb_service.load("SGD_dataset_TINY")
    assert not df.empty, "Dataframe is empty"
    state_tracker = DiaStateTracker()
    column_for_intentions = 'Intention'
    column_for_actions = 'Action'
    df = state_tracker.get_state_and_actions(
        df,
        column_for_intentions=column_for_intentions,
        column_for_actions=column_for_actions
    )
    #df.to_csv('SGD_dataset_TINY_state_tracker.csv', index=False)

    """mongodb_service.save(df, f"SGD_dataset_TINY_state_tracker_{column_for_intentions}_{column_for_actions}_"
                             f"max_history={max_history_length}")"""
    embeddings = np.array(df['State'].tolist())
    print(embeddings.shape)
    print(embeddings[0][0])
    print(embeddings)


if __name__ == '__main__':
    main()
