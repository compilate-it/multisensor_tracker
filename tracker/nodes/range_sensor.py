import numpy as np


class RangeSensor:

    #Range-only sensor.
    #Measures: z = range + noise
    #State: [px, py, vx, vy]


    def __init__(self, sigma_r_m: float):
        self.sigma_r_m = sigma_r_m

    # --------------------------------------------------
    # Generate noisy measurement
    # --------------------------------------------------

    def measure(
        self,
        target_state: np.ndarray,
        sensor_position: np.ndarray
    ) -> dict:

        true_range = self.measurement_function(
            target_state,
            sensor_position
        )[0]

        noise = np.random.normal(
            0.0,
            self.sigma_r_m
        )

        measured_range = true_range + noise

        return {
            "type": "range",
            "z": np.array([measured_range]),
            "R": self.measurement_covariance()
        }

    # --------------------------------------------------
    # h(x)
    # --------------------------------------------------

    def measurement_function(
        self,
        state: np.ndarray,
        sensor_position: np.ndarray
    ) -> np.ndarray:

        px, py = state[0], state[1]
        sx, sy = sensor_position

        dx = px - sx
        dy = py - sy

        r = np.sqrt(dx**2 + dy**2)

        return np.array([r])

    # --------------------------------------------------
    # H(x)
    # --------------------------------------------------

    def jacobian(
        self,
        state: np.ndarray,
        sensor_position: np.ndarray
    ) -> np.ndarray:

        px, py = state[0], state[1]
        sx, sy = sensor_position

        dx = px - sx
        dy = py - sy

        r = np.sqrt(dx**2 + dy**2)

        if r < 1e-12:
            r = 1e-12

        return np.array([
            [
                dx / r,
                dy / r,
                0.0,
                0.0
            ]
        ])

    # --------------------------------------------------
    # R
    # --------------------------------------------------

    def measurement_covariance(self):

        return np.array([
            [self.sigma_r_m**2]
        ])

    # --------------------------------------------------
    # Innovation
    # --------------------------------------------------

    @staticmethod
    def innovation(
        z: np.ndarray,
        z_pred: np.ndarray
    ) -> np.ndarray:

        return z - z_pred