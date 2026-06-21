# Target motion model - ground truth (true target state)

import numpy as np


class Target:

    #Ground-truth target.
    #State matrix:[px, py, vx, vy]

    def __init__(
        self,
        initial_state: np.ndarray,
        dt: float
    ):
        self.state = np.asarray(
            initial_state,
            dtype=float
        )

        self.dt = dt

    def step(self):

        px, py, vx, vy = self.state

        px += vx * self.dt
        py += vy * self.dt

        self.state = np.array([
            px,
            py,
            vx,
            vy
        ])

        return self.state.copy()

    def get_state(self):
        return self.state.copy()