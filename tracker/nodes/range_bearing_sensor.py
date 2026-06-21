import numpy as np


class RangeBearingSensor:

    #Range-bearing sensor.
    #Measures: z = [range, bearing]
    #State: [px, py, vx, vy]


    def __init__(
        self,
        sigma_r_m: float,
        sigma_az_deg: float
    ):
        self.sigma_r_m = sigma_r_m

        self.sigma_az_deg = sigma_az_deg
        self.sigma_az_rad = np.deg2rad(
            sigma_az_deg
        )

    # --------------------------------------------------
    # Generate noisy measurement
    # --------------------------------------------------

    def measure(
        self,
        target_state: np.ndarray,
        sensor_position: np.ndarray
    ) -> dict:

        true_measurement = self.measurement_function(
            target_state,
            sensor_position
        )

        measured_range = (
            true_measurement[0]
            + np.random.normal(
                0.0,
                self.sigma_r_m
            )
        )

        measured_bearing = (
            true_measurement[1]
            + np.random.normal(
                0.0,
                self.sigma_az_rad
            )
        )

        measured_bearing = self.normalize_angle(
            measured_bearing
        )

        return {
            "type": "range_bearing",
            "z": np.array([
                measured_range,
                measured_bearing
            ]),
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

        bearing = np.arctan2(
            dy,
            dx
        )

        return np.array([
            r,
            bearing
        ])

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

        r2 = dx**2 + dy**2

        if r2 < 1e-12:
            r2 = 1e-12

        r = np.sqrt(r2)

        return np.array([
            [
                dx / r,
                dy / r,
                0.0,
                0.0
            ],
            [
                -dy / r2,
                dx / r2,
                0.0,
                0.0
            ]
        ])

    # --------------------------------------------------
    # R
    # --------------------------------------------------

    def measurement_covariance(self):

        return np.array([
            [
                self.sigma_r_m**2,
                0.0
            ],
            [
                0.0,
                self.sigma_az_rad**2
            ]
        ])

    # --------------------------------------------------
    # Innovation
    # --------------------------------------------------

    def innovation(
        self,
        z: np.ndarray,
        z_pred: np.ndarray
    ) -> np.ndarray:

        innovation = z - z_pred

        innovation[1] = self.normalize_angle(
            innovation[1]
        )

        return innovation

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    @staticmethod
    def normalize_angle(
        angle: float
    ) -> float:

        return (
            (angle + np.pi)
            % (2 * np.pi)
            - np.pi
        )
