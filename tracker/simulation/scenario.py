# Trajectory generation

import numpy as np


class Scenario:

    #Straight line trajectory:
    @staticmethod
    def straight_line(
        initial_state,
        dt,
        steps
    ):

        trajectory = []

        state = np.asarray(
            initial_state,
            dtype=float
        ).copy()

        for _ in range(steps):

            trajectory.append(
                state.copy()
            )

            state[0] += state[2] * dt
            state[1] += state[3] * dt

        return np.array(trajectory)

    @staticmethod
    def circular(
        center,
        radius,
        angular_velocity,
        dt,
        steps
    ):

        #Circular trajectory.

        trajectory = []

        for k in range(steps):

            t = k * dt

            x = center[0] + radius * np.cos(
                angular_velocity * t
            )

            y = center[1] + radius * np.sin(
                angular_velocity * t
            )

            vx = (
                -radius
                * angular_velocity
                * np.sin(
                    angular_velocity * t
                )
            )

            vy = (
                radius
                * angular_velocity
                * np.cos(
                    angular_velocity * t
                )
            )

            trajectory.append([
                x,
                y,
                vx,
                vy
            ])

        return np.array(trajectory)

    @staticmethod
    def random_walk(
        initial_state,
        dt,
        steps,
        velocity_noise=0.5
    ):

        #Random walk trajectory.

        trajectory = []

        state = np.asarray(
            initial_state,
            dtype=float
        ).copy()

        for _ in range(steps):

            trajectory.append(
                state.copy()
            )

            state[2] += np.random.normal(
                0,
                velocity_noise
            )

            state[3] += np.random.normal(
                0,
                velocity_noise
            )

            state[0] += state[2] * dt
            state[1] += state[3] * dt

        return np.array(trajectory)