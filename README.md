# Multisensor tracker

This is a simulation of a distributed target tracking system using multiple heterogeneous sensor nodes and an Extended Kalman Filter (EKF) for data fusion.<br>
How to start:<br>

1. Install dependencies:<br>
pip install numpy pyyaml

2. Running the simulation:<br>
   Execute the main.py script from the project root:<br>
   python src/main.py

3. Configuration:<br>
   The entire project is configred through yaml files.<br>
   a. Sensor config: configs/nodes.yaml<br>
   b. Simulation config (including EKF params): configs/simulation.yaml<br>
   c. Scenario: configs/scenarios/circular.yaml (circular target trajectory)
