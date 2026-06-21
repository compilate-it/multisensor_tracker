
#Sensor implements the Node interface
# measurement model h(x)
# Jacobian H(x)
# covariance R
# angle wrapping

import numpy as np


class BearingSensor:

    #Bearing-only sensor.
    #Measures: z = atan2(y - ys, x - xs) + noise
    #State vector:[x, y, vx, vy]

    def __init__(self, sigma_az_deg: float):

        #Parameters
        #sigma_az_deg : float
        #Bearing noise standard deviation in degrees.

        self.sigma_az_deg = sigma_az_deg
        self.sigma_az_rad = np.deg2rad(sigma_az_deg)

    # --------------------------------------------------
    # Generate noisy measurement
    # --------------------------------------------------
    def measure(
        self,
        target_state: np.ndarray,
        sensor_position: np.ndarray
    ) -> dict:

        #Generate noisy bearing measurement.
        #Returns
        # dict
        #     {
        #         "type": "bearing",
        #         "z": np.array([bearing]),
        #         "R": covariance matrix
        #     }


        true_bearing = self.measurement_function(
            target_state,
            sensor_position
        )[0]

        noise = np.random.normal(
            0.0,
            self.sigma_az_rad
        )

        measured_bearing = self.normalize_angle(
            true_bearing + noise
        )

        return {
            "type": "bearing",
            "z": np.array([measured_bearing]),
            "R": self.measurement_covariance()
        }

    # --------------------------------------------------
    # Nonlinear measurement model h(x)
    # Mathematical model of the bearing sensor
    # --------------------------------------------------
    def measurement_function(
        self,
        state: np.ndarray,
        sensor_position: np.ndarray
    ) -> np.ndarray:

        # Computes expected bearing measurement.
        # h(x) = atan2(dy, dx)
        # Returns:
        # np.ndarray shape (1,)

        px, py = state[0], state[1]

        sx, sy = sensor_position

        dx = px - sx
        dy = py - sy

        bearing = np.arctan2(dy, dx)

        return np.array([bearing])

    # --------------------------------------------------
    # Jacobian H(x)
    # --------------------------------------------------
    def jacobian(
        self,
        state: np.ndarray,
        sensor_position: np.ndarray
    ) -> np.ndarray:

        # Jacobian of bearing measurement model.
        #
        # H = [ -dy/r²   dx/r²   0   0 ]

        px, py = state[0], state[1]

        sx, sy = sensor_position

        dx = px - sx
        dy = py - sy

        r2 = dx**2 + dy**2

        if r2 < 1e-12:
            r2 = 1e-12

        H = np.array([
            [
                -dy / r2,
                 dx / r2,
                 0.0,
                 0.0
            ]
        ])

        return H

    # --------------------------------------------------
    # Measurement covariance
    # --------------------------------------------------
    def measurement_covariance(self) -> np.ndarray:

        #Covariance matrix for EKF.


        return np.array([
            [self.sigma_az_rad**2]
        ])

    # --------------------------------------------------
    # Innovation
    # --------------------------------------------------
    def innovation(
        self,
        z: np.ndarray,
        z_pred: np.ndarray
    ) -> np.ndarray:

        #Computes innovation with angle wrapping.
        #y = z - h(x)

        innovation = z - z_pred

        innovation[0] = self.normalize_angle(
            innovation[0]
        )

        return innovation

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------
    @staticmethod
    def normalize_angle(angle: float) -> float:

        #Wrap angle to [-pi, pi].

        return (angle + np.pi) % (2 * np.pi) - np.pi