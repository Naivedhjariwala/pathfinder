# AI-Powered Traffic Management System

## Overview

This project is a software prototype for an AI-based traffic management system, developed as a solution for a hackathon challenge. The goal is to reduce urban traffic congestion by optimizing traffic signal timings.

The system uses a Deep Q-Network (DQN), a Reinforcement Learning algorithm, to control traffic lights in a simulated four-way intersection. The agent learns from real-time traffic data within the simulation to make intelligent decisions, aiming to minimize vehicle commute times.

## Features

- **SUMO Simulation:** A detailed simulation of a four-way intersection built using [Eclipse SUMO](https://www.eclipse.org/sumo/), a microscopic traffic simulation suite.
- **Reinforcement Learning Agent:** A DQN agent implemented from scratch in Python using the PyTorch library. The agent observes the state of the intersection (queue lengths) and takes actions (switching or maintaining traffic light phases) to optimize traffic flow.
- **Real-time Data API:** A lightweight Flask web server that runs concurrently with the simulation, providing a JSON API endpoint (`/data`) with live data from the intersection.
- **Live Dashboard:** A web-based dashboard (`index.html`) that visualizes real-time data from the API, including queue lengths on each lane and the current traffic light phase.
- **Performance Evaluation:** A script (`evaluate.py`) to quantitatively measure the effectiveness of the RL agent by comparing its average vehicle commute time against a traditional fixed-timer baseline.

## System Requirements

- **SUMO:** The core traffic simulation suite.
- **Python 3.8+**
- Python libraries as listed in `requirements.txt`.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install SUMO:**
    On Debian-based systems (like Ubuntu), you can install SUMO and its command-line tools using `apt-get`.
    ```bash
    sudo apt-get update
    sudo apt-get install sumo sumo-tools sumo-doc
    ```
    For other operating systems, please refer to the [official SUMO installation guide](https://sumo.dlr.de/docs/Installing/index.html).

3.  **Install Python Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## How to Run the Prototype

1.  **Start the Training and Dashboard Server:**
    Run the main Python script. This will start the SUMO simulation in the background, train the RL agent, and launch the Flask web server for the dashboard.
    ```bash
    python3 main.py
    ```
    You will see output in your terminal for each training episode. The server will be running at `http://0.0.0.0:5000`.

2.  **View the Live Dashboard:**
    Open your web browser and navigate to:
    [http://localhost:5000](http://localhost:5000)

    The dashboard will automatically connect to the backend and start displaying real-time data from the simulation.

3.  **Evaluate Performance:**
    After the training script (`main.py`) has run for at least one full episode, it will generate a `tripinfo.xml` file. To compare the agent's performance against the fixed-timer baseline, run the evaluation script:
    ```bash
    python3 evaluate.py
    ```
    This will output the average commute time for both the baseline and the RL agent, along with the percentage improvement.

## Code Structure and "How It Works"

-   `main.py`: This is the central script of the project.
    -   **Flask App:** Sets up the web server and the `/` and `/data` API endpoints.
    -   **RL Agent:** Contains the classes for the DQN agent:
        -   `QNetwork`: A PyTorch neural network model that learns to predict the value of actions.
        -   `ReplayBuffer`: Stores past experiences (state, action, reward, next_state) for the agent to learn from.
        -   `Agent`: The main agent class that encapsulates the network and buffer, and handles the act-learn cycle.
    -   `train()`: The main training loop that orchestrates the entire process. It runs the simulation for multiple episodes, gets actions from the agent, and updates the shared data for the dashboard.

-   **SUMO Files (`*.xml`)**
    -   `intersection.net.xml`: Defines the road network (nodes and edges).
    -   `intersection.rou.xml`: Defines the vehicle types, routes, and traffic flow rates.
    -   `intersection.sumocfg`: The main simulation configuration file, which brings the network and route files together.
    -   `fixed_timer.add.xml`: An additional file defining the traffic light logic for the fixed-timer baseline.
    -   `intersection_fixed.sumocfg`: A separate configuration to run the baseline simulation.

-   `static/index.html`: The frontend dashboard. It uses JavaScript's `fetch` API to periodically poll the `/data` endpoint and update the charts and metrics using Chart.js.

-   `evaluate.py`: This script parses the `tripinfo.xml` files generated by the agent and baseline simulations. It uses Python's built-in XML parser to calculate the average trip duration for all vehicles and computes the percentage improvement.

## Current Performance & Future Work

After a training run of 50 episodes, the agent currently achieves a **~3% improvement** in average commute time over a standard fixed-timer traffic light.

To reach the 10% target outlined in the hackathon challenge, further work is needed:
-   **Extended Training:** The agent would benefit significantly from a much longer training period (e.g., 500+ episodes).
-   **Hyperparameter Tuning:** The learning rate, discount factor (gamma), and epsilon decay schedule could be tuned for better performance.
-   **State/Reward Engineering:** The state and reward definitions are currently simple for this prototype. More complex representations (e.g., including vehicle speed, or a more sophisticated reward function) could lead to better results.


# Don't trust Vector's code vector is an idiot