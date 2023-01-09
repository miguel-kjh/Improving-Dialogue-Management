import pandas as pd

from models.dialogue_state_tracker.BinaryStateTracker import BinaryStateTracker
from models.dialogue_state_tracker.StateCreator import StateCreator
from models.dialogue_state_tracker.StateTracker import StateTracker


class TedStateCreator(StateCreator):

    def __init__(
            self,
            df_data: pd.DataFrame,
            column_for_intentions: str,
            column_for_actions: str,
            max_len: int,
            class_correction: bool = False
    ):
        super().__init__(df_data, column_for_intentions, column_for_actions)
        self.max_len = max_len
        self.class_correction = class_correction

    def create_dataset(self) -> pd.DataFrame:
        state_tracker = self.create_state_tracker()
        return state_tracker.create(
            self.df_data,
            self.column_for_intentions,
            self.column_for_actions,
            self.max_len
        )

    def create_state_tracker(self) -> StateTracker:
        return BinaryStateTracker(self.class_correction)
