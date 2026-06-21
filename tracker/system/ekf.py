import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for 2D target tracking.

    State:
        x = [px, py, vx, vy]^T
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        dt: float,
        process_noise_std: float
    ):
        self.x = initial_state.astype(float)
        self.P = initial_covariance.astype(float)

        self.dt = dt

        self.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        q = process_noise_std ** 2

        self.Q = q * np.array([
            [dt**4/4, 0,       dt**3/2, 0],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0],
            [0,       dt**3/2, 0,       dt**2]
        ])

    # -------------------------------------------------
    # Prediction step
    # -------------------------------------------------

    def predict(self):

        #EKF prediction.
        #x = F x
        #P = F P F^T + Q


        self.x = self.F @ self.x

        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q
        )

    # -------------------------------------------------
    # Generic EKF update
    # -------------------------------------------------
    #In: z : measurement vector
    # h(x, sensor_position) - measurement function
    # H(x, sensor_position) - Jacobian for linearizing the sensor model
    # R matrix - measurement covariance (sensor uncertainty)
    # sensor position
    # additional function for angle wrapping (in bearing sensors)

    def update(
        self,
        z: np.ndarray,
        measurement_function,
        jacobian_function,
        measurement_covariance: np.ndarray,
        sensor_position: np.ndarray,
        innovation_function=None
    ):

        # Predicted measurement
        z_pred = measurement_function(
            self.x,
            sensor_position
        )

        # Jacobian
        H = jacobian_function(
            self.x,
            sensor_position
        )

        # Innovation
        if innovation_function is None:
            y = z - z_pred
        else:
            y = innovation_function(
                z,
                z_pred
            )

        # Innovation covariance
        S = (
            H @ self.P @ H.T
            + measurement_covariance
        )

        # Kalman gain
        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(len(self.x))

        self.P = (
            I - K @ H
        ) @ self.P

    # -------------------------------------------------
    # Utility methods
    # -------------------------------------------------

    @property
    def position(self):
        return self.x[:2]

    @property
    def velocity(self):
        return self.x[2:]

    def get_state(self):
        return self.x.copy()

    def get_covariance(self):
        return self.P.copy()