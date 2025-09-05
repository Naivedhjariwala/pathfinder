import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
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
    app.run(host='0.0.0.0', port=5000, debug=False)

sumoBinary = "sumo" 
sumoCmd = [sumoBinary, "-c", "intersection.sumocfg"]

TRAFFIC_LIGHT_ID = "C"
INCOMING_LANES = ["N_to_C_0", "N_to_C_1", "S_to_C_0", "S_to_C_1", "E_to_C_0", "E_to_C_1", "W_to_C_0", "W_to_C_1"]
PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2

STATE_SIZE = len(INCOMING_LANES)
ACTION_SIZE = 2

BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1e-3)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def get_state():
    state = [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in INCOMING_LANES]
    return np.array(state)

def get_reward(current_state_queues):
    return -np.sum(current_state_queues)

def train(n_episodes=10, max_t=3600, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=0)

    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()

    for i_episode in range(1, n_episodes+1):
        traci.start(sumoCmd)
        state = get_state()
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)

            current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
            if (action == 1 and (current_phase == PHASE_NS_GREEN or current_phase == PHASE_EW_GREEN)):
                new_phase = PHASE_EW_GREEN if current_phase == PHASE_NS_GREEN else PHASE_NS_GREEN
                traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, new_phase)

            simulation_steps_per_decision = 10
            for _ in range(simulation_steps_per_decision):
                traci.simulationStep()

            next_state = get_state()
            reward = get_reward(next_state)
            done = traci.simulation.getMinExpectedNumber() == 0
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            with data_lock:
                simulation_data['episode'] = i_episode
                simulation_data['time_step'] = t * simulation_steps_per_decision
                simulation_data['state'] = state.tolist()
                simulation_data['reward'] = float(reward)
                simulation_data['current_phase'] = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
                simulation_data['average_score'] = float(np.mean(scores_window)) if scores_window else 0.0

            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if i_episode % 10 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        traci.close()

    return scores

if __name__ == "__main__":
    scores = train(n_episodes=10)
    print("Training complete.")