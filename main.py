import os
import sys

# Check for SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    # Adding a default path for the sandbox environment
    # This might need adjustment if the path is different
    default_sumo_tools_path = '/usr/share/sumo/tools'
    if os.path.exists(default_sumo_tools_path):
        sys.path.append(default_sumo_tools_path)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME' or ensure SUMO tools are in a standard path.")

import traci

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
from flask import Flask, jsonify, send_from_directory
import threading
import time

# --- Web Server Setup ---
app = Flask(__name__, static_folder='static')
simulation_data = {}
data_lock = threading.Lock()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/data')
def get_data():
    with data_lock:
        return jsonify(simulation_data)

def start_web_server():
    # Running in debug mode is not recommended for production
    # but is fine for this hackathon context.
    app.run(host='0.0.0.0', port=5000, debug=False)

# --- End Web Server Setup ---


# SUMO configuration
sumoBinary = "sumo" # or "sumo-gui" for visual
sumoCmd = [sumoBinary, "-c", "intersection.sumocfg"]

# RL Agent Constants
TRAFFIC_LIGHT_ID = "C"
INCOMING_LANES = ["N_to_C_0", "N_to_C_1", "S_to_C_0", "S_to_C_1", "E_to_C_0", "E_to_C_1", "W_to_C_0", "W_to_C_1"]
# Phases: 0 = NS Green, 1 = NS Yellow, 2 = EW Green, 3 = EW Yellow
PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2

# State: Queue length on each of the 8 incoming lanes
STATE_SIZE = len(INCOMING_LANES)
# Action: 0 = Keep current phase, 1 = Switch to the other green phase
ACTION_SIZE = 2

# DQN Agent Hyperparameters
BUFFER_SIZE = 10000  # Replay buffer size
BATCH_SIZE = 32      # Minibatch size
GAMMA = 0.99         # Discount factor
LR = 5e-4            # Learning rate
UPDATE_EVERY = 4     # How often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object."""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def get_state():
    """
    Get the number of halting vehicles on each incoming lane.
    """
    state = [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in INCOMING_LANES]
    return np.array(state)

def get_reward(current_state_queues):
    """
    Calculate reward as the negative sum of waiting cars.
    """
    return -np.sum(current_state_queues)

def train(n_episodes=10, max_t=3600, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=0)

    with data_lock:
        simulation_data['scores_history'] = []

    # Start the web server in a background thread
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()

    for i_episode in range(1, n_episodes+1):
        traci.start(sumoCmd)

        state = get_state()
        score = 0

        # Using max_t as a safeguard against infinitely running episodes
        for t in range(max_t):
            action = agent.act(state, eps)

            # Execute action and let simulation run
            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            if (action == 1 and (current_phase == PHASE_NS_GREEN or current_phase == PHASE_EW_GREEN)):
                new_phase = PHASE_EW_GREEN if current_phase == PHASE_NS_GREEN else PHASE_NS_GREEN
                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, new_phase)

            # Step the simulation forward
            simulation_steps_per_decision = 10
            for _ in range(simulation_steps_per_decision):
                traci.simulationStep()

            next_state = get_state()
            reward = get_reward(next_state)
            done = traci.simulation.getMinExpectedNumber() == 0
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            # Update shared data for the web server
            with data_lock:
                simulation_data['episode'] = i_episode
                simulation_data['time_step'] = t * simulation_steps_per_decision
                simulation_data['state'] = state.tolist()
                simulation_data['reward'] = float(reward)
                simulation_data['current_phase'] = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                simulation_data['average_score'] = float(np.mean(scores_window)) if scores_window else 0.0

            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        with data_lock:
            simulation_data['scores_history'].append(np.mean(scores_window))

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if i_episode % 10 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

        traci.close()

    return scores

if __name__ == "__main__":
    # For a hackathon, we'd run a small number of episodes.
    # For a real project, this would be much larger.
    scores = train(n_episodes=10)
    print("Training complete.")
    print("Dashboard is still running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
