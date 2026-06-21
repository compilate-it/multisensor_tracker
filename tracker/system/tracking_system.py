import numpy as np


class TrackingSystem:

    #- Collect measurements from all nodes, run EKF and sequential fusion, store history

    def __init__(self, nodes, ekf):
        self.nodes = nodes
        self.ekf = ekf

        self.history = {
            "time": [],
            "truth": [],
            "estimate": [],
            "covariance": [],
            "measurements": []
        }

        self.step_count = 0

    # --------------------------------------------------
    # Single simulation step
    # --------------------------------------------------

    def step(self, truth_state):

        #One tracking cycle. Managed by the engine
        #In: truth_state (ground truth target state), ndarray
        #Out: Estimated state (ndarray)

        # 1. Collect measurements
        measurements = []

        for node in self.nodes:

            measurement = node.measure(
                truth_state
            )

            measurements.append(
                {
                    "node": node,
                    "measurement": measurement
                }
            )

        # 2. Predict
        self.ekf.predict()

        # 3. Fuse all measurements
        for entry in measurements:

            node = entry["node"]
            measurement = entry["measurement"]

            sensor = node.sensor

            self.ekf.update(
                z=measurement["z"],
                measurement_function=sensor.measurement_function,
                jacobian_function=sensor.jacobian,
                measurement_covariance=measurement["R"],
                sensor_position=node.position,
                innovation_function=getattr(
                    sensor,
                    "innovation",
                    None
                )
            )

        # 4. Save history
        self._store_step(
            truth_state,
            measurements
        )

        self.step_count += 1

        return self.ekf.get_state()

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    def _store_step(
        self,
        truth_state,
        measurements
    ):

        self.history["time"].append(
            self.step_count
        )

        self.history["truth"].append(
            truth_state.copy()
        )

        self.history["estimate"].append(
            self.ekf.get_state()
        )

        self.history["covariance"].append(
            self.ekf.get_covariance()
        )

        self.history["measurements"].append(
            measurements
        )

    # --------------------------------------------------
    # Accessors
    # --------------------------------------------------

    def get_estimate(self):
        return self.ekf.get_state()

    def get_covariance(self):
        return self.ekf.get_covariance()

    def get_history(self):
        return self.history

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    def compute_position_errors(self):

        errors = []

        for truth, estimate in zip(
            self.history["truth"],
            self.history["estimate"]
        ):

            error = np.linalg.norm(
                truth[:2] - estimate[:2]
            )

            errors.append(error)

        return np.array(errors)

    def compute_rmse(self):

        errors = self.compute_position_errors()

        return np.sqrt(
            np.mean(errors**2)
        )

    # --------------------------------------------------
    # Covariance diagnostics
    # --------------------------------------------------

    def covariance_trace(self):

        traces = []

        for P in self.history["covariance"]:

            traces.append(
                np.trace(P[:2, :2])
            )

        return np.array(traces)