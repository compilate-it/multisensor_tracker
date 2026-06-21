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

        for step, truth_state in enumerate(self.trajectory):

            estimate = self.system.step(
                truth_state
            )

            uncertainty = (
                self.system.get_position_uncertainty()
            )

            print(
                f"[{step:04d}] "
                f"Pos=({estimate[0]:8.2f}, {estimate[1]:8.2f}) "
                f"Vel=({estimate[2]:6.2f}, {estimate[3]:6.2f}) "
                f"σ_pos={uncertainty:6.2f} m"
            )
        return self.system.get_history()