import numpy as np
class Node:

    def __init__(
        self,
        node_id,
        position,
        sensor
    ):
        self.id = node_id
        self.position = np.asarray(position)
        self.sensor = sensor

    def measure(self, target_state):
        return self.sensor.measure(
            target_state,
            self.position
        )