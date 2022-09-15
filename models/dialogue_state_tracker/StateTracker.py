from abc import ABC
import pandas as pd
import numpy as np


class StateTracker(ABC):

    def __init__(self, class_correction: bool = False):
        self.class_correction = class_correction
        self._mandatory_slot_column = 'Mandatory Slots'
        self._mandatory_slot_column_value = 'Mandatory Slots Value'
        self._optional_slot_column = 'Optional Slots'
        self._optional_slot_column_value = 'Optional Slots Value'

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

    @staticmethod
    def _check_mx_history_management(mx_history_length: int) -> int:
        assert mx_history_length > 0, "mx_history_length must be greater than 0"

        if mx_history_length is None:
            mx_history_length = 1

        return mx_history_length

    def _change_fuzzy_action(self, action: str, is_mandatory_slots) -> str:
        if self.class_correction:
            if is_mandatory_slots and (action == 'REQ_MORE' or action == 'CONFIRM'):
                return 'CONFIRM'

        return action

    def get_state_and_actions(
            self,
            df_data_oring: pd.DataFrame,
            column_for_intentions,
            column_for_actions,
            mx_history_length=None
    ) -> pd.DataFrame:
        pass
