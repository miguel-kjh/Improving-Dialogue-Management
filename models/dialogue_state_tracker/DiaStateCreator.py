import pandas as pd

from models.dialogue_state_tracker.DiaStateTracker import DiaStateTracker
from models.dialogue_state_tracker.StateCreator import StateCreator
from models.dialogue_state_tracker.StateTracker import StateTracker


class DiaStateCreator(StateCreator):

    def __init__(
            self,
            df_data: pd.DataFrame,
            column_for_intentions: str,
            column_for_actions: str
    ):
        super().__init__(df_data, column_for_intentions, column_for_actions)

    def create_state_tracker(self) -> StateTracker:
        return DiaStateTracker()