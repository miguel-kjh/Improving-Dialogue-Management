from abc import abstractmethod, ABC

import pandas as pd

from models.dialogue_state_tracker.StateTracker import StateTracker


class StateCreator(ABC):

    def __init__(self, df_data: pd.DataFrame, column_for_intentions: str, column_for_actions: str):
        self.df_data = df_data
        self.column_for_intentions = column_for_intentions
        self.column_for_actions = column_for_actions

    def create_dataset(self) -> pd.DataFrame:
        state_tracker = self.create_state_tracker()
        return state_tracker.create(
            self.df_data,
            self.column_for_intentions,
            self.column_for_actions
        )

    @abstractmethod
    def create_state_tracker(self) -> StateTracker:
        pass
