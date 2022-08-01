import numpy as np


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
