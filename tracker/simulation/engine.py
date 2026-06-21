# Main simulation loop

import numpy as np


class SimulationEngine:

    def __init__(
        self,
        tracking_system,
        trajectory
    ):
        self.system = tracking_system
        self.trajectory = trajectory

    def run(self):

        for truth_state in self.trajectory:

            self.system.step(
                truth_state
            )

        return self.system.get_history()