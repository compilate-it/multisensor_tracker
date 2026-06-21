from utils.config_parser import load_yaml

from utils.factories import (
    create_nodes,
    create_ekf,
    create_trajectory
)

from system.tracking_system import (
    TrackingSystem
)

from simulation.engine import (
    SimulationEngine
)


nodes_cfg = load_yaml(
    "configs/nodes.yaml"
)

sim_cfg = load_yaml(
    "configs/simulation.yaml"
)

scenario_cfg = load_yaml(
    "configs/scenarios/circular.yaml"
)


nodes = create_nodes(
    nodes_cfg
)

ekf = create_ekf(
    sim_cfg
)

trajectory = create_trajectory(
    sim_cfg,
    scenario_cfg
)

system = TrackingSystem(
    nodes,
    ekf
)

engine = SimulationEngine(
    system,
    trajectory
)

history = engine.run()

print("Simulation finished")
print(f"Steps: {len(history['time'])}")
print(f"RMSE: {system.compute_rmse():.3f} m")

final_estimate = system.get_estimate()

print("\nFinal estimate:")
print(f"x  = {final_estimate[0]:.2f}")
print(f"y  = {final_estimate[1]:.2f}")
print(f"vx = {final_estimate[2]:.2f}")
print(f"vy = {final_estimate[3]:.2f}")