import numpy as np

from nodes.node import Node
from nodes.bearing_sensor import BearingSensor
from nodes.range_sensor import RangeSensor
from nodes.range_bearing_sensor import RangeBearingSensor


def create_nodes(config):

    nodes = []

    for node_cfg in config["nodes"]:

        sensor_type = node_cfg["type"]

        if sensor_type == "bearing":

            sensor = BearingSensor(
                sigma_az_deg=node_cfg["sigma_az_deg"]
            )

        elif sensor_type == "range":

            sensor = RangeSensor(
                sigma_r_m=node_cfg["sigma_r_m"]
            )

        elif sensor_type == "range_bearing":

            sensor = RangeBearingSensor(
                sigma_r_m=node_cfg["sigma_r_m"],
                sigma_az_deg=node_cfg["sigma_az_deg"]
            )

        else:
            raise ValueError(
                f"Unknown sensor type: {sensor_type}"
            )

        nodes.append(
            Node(
                node_id=node_cfg["id"],
                position=np.array(
                    node_cfg["position"]
                ),
                sensor=sensor
            )
        )

    return nodes


import numpy as np

from system.ekf import ExtendedKalmanFilter


def create_ekf(config):

    ekf_cfg = config["ekf"]

    return ExtendedKalmanFilter(
        initial_state=np.array(
            ekf_cfg["initial_state"]
        ),
        initial_covariance=np.array(
            ekf_cfg["initial_covariance"]
        ),
        dt=config["simulation"]["dt"],
        process_noise_std=ekf_cfg[
            "process_noise_std"
        ]
    )


from simulation.scenario import Scenario


def create_trajectory(
    sim_cfg,
    scenario_cfg
):

    scenario_type = sim_cfg[
        "scenario"
    ]["type"]

    dt = sim_cfg[
        "simulation"
    ]["dt"]

    steps = sim_cfg[
        "simulation"
    ]["steps"]

    if scenario_type == "circular":

        return Scenario.circular(
            center=scenario_cfg["center"],
            radius=scenario_cfg["radius"],
            angular_velocity=scenario_cfg[
                "angular_velocity"
            ],
            dt=dt,
            steps=steps
        )

    raise ValueError(
        f"Unsupported scenario {scenario_type}"
    )