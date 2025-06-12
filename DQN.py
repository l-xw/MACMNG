import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.autograd.set_detect_anomaly(True)


class Net(nn.Module):
    def __init__(self, floors_number, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        self.floors_number = floors_number
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(n_states, n_hidden))
        for i in range(0, floors_number - 2):
            self.linears.append(nn.Linear(n_hidden, n_hidden))
        self.linears.append(nn.Linear(n_hidden, n_actions))

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = F.leaky_relu(layer(x))
        x = self.linears[-1](x)
        return x


class DQN:
    def __init__(self, floors_number, n_states, n_hidden, n_actions, learning_rate, gamma, epsilon, target_update, device):
        self.floors_number = floors_number
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device

        self.count = 0

        self.q_net = Net(self.floors_number, self.n_states, self.n_hidden, self.n_actions).to(device)
        self.target_q_net = Net(self.floors_number, self.n_states, self.n_hidden, self.n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1
        return dqn_loss.item()

    def get_action(self, state, legitimate):
        if np.random.random() < 1 - self.epsilon:
            action = np.random.choice([index for index, data in enumerate(legitimate) if data])
        else:
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            q_values = self.q_net(state)
            illegal_mask = torch.tensor([float('-inf') if not leg else 0 for leg in legitimate], device=self.device)
            q_values = q_values + illegal_mask
            action = q_values.argmax().item()
        return action


class MODQN:
    def __init__(self, floors_number, n_states, n_hidden, n_actions, learning_rate, gamma, epsilon, target_update, device):
        self.floors_number = floors_number
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device

        self.MK_W = 1

        self.MK_q_net = Net(self.floors_number, self.n_states, self.n_hidden, self.n_actions).to(device)
        self.MK_target_q_net = Net(self.floors_number, self.n_states, self.n_hidden, self.n_actions).to(device)
        self.MK_optimizer = torch.optim.Adam(self.MK_q_net.parameters(), lr=self.learning_rate)

        self.TEC_q_net = Net(self.floors_number, self.n_states, self.n_hidden, self.n_actions).to(device)
        self.TEC_target_q_net = Net(self.floors_number, self.n_states, self.n_hidden, self.n_actions).to(device)
        self.TEC_optimizer = torch.optim.Adam(self.TEC_q_net.parameters(), lr=self.learning_rate)

        self.count = 0

    def update(self, transition_dict):
        MK_state = []
        TEC_state = []
        state = transition_dict['states']
        for s in state:
            MK_state.append(s[0])
            TEC_state.append(s[1])
        MK_state = torch.tensor(np.array(MK_state), dtype=torch.float).to(self.device)
        TEC_state = torch.tensor(np.array(TEC_state), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(
            self.device)
        rewards = list(transition_dict['rewards'])

        MK_reward = torch.tensor(np.array([x[0] for x in rewards]), dtype=torch.float).view(-1, 1).to(self.device)
        TEC_reward = torch.tensor(np.array([x[1] / 10 for x in rewards]), dtype=torch.float).view(-1, 1).to(self.device)

        MK_next_state = []
        TEC_next_state = []
        next_states = transition_dict['next_states']
        for s in next_states:
            MK_next_state.append(s[0])
            TEC_next_state.append(s[1])
        MK_next_state = torch.tensor(np.array(MK_next_state), dtype=torch.float).to(self.device)
        TEC_next_state = torch.tensor(np.array(TEC_next_state), dtype=torch.float).to(self.device)

        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        MK_q_values = self.MK_q_net(MK_state).gather(1, actions)
        TEC_q_values = self.TEC_q_net(TEC_state).gather(1, actions)

        MK_max_action = self.MK_q_net(MK_next_state)
        TEC_max_action = self.TEC_q_net(TEC_next_state)

        max_action = (self.MK_W * F.softmax(MK_max_action, dim=1) + (1 - self.MK_W) * F.softmax(TEC_max_action, dim=1)).max(1)[1].view(-1, 1)

        MK_max_next_q_values = self.MK_target_q_net(MK_next_state).gather(1, max_action)
        MK_q_targets = MK_reward + self.gamma * MK_max_next_q_values * (1 - dones)
        MK_dqn_loss = torch.mean(F.mse_loss(MK_q_values, MK_q_targets))

        TEC_max_next_q_values = self.TEC_target_q_net(TEC_next_state).gather(1, max_action)
        TEC_q_targets = TEC_reward + self.gamma * TEC_max_next_q_values * (1 - dones)
        TEC_dqn_loss = torch.mean(F.mse_loss(TEC_q_values, TEC_q_targets))

        loss = self.MK_W * MK_dqn_loss + (1 - self.MK_W) * TEC_dqn_loss

        self.MK_optimizer.zero_grad()
        self.TEC_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.MK_q_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.TEC_q_net.parameters(), max_norm=1.0)

        self.MK_optimizer.step()
        self.TEC_optimizer.step()

        if self.count % self.target_update == 0:
            self.MK_target_q_net.load_state_dict(
                self.MK_q_net.state_dict())
            self.TEC_target_q_net.load_state_dict(
                self.TEC_q_net.state_dict())
        self.count += 1
        return loss.item()

    def get_action(self, state, legitimate):
        if np.random.random() < 1 - self.epsilon:
            action = np.random.choice([index for index, data in enumerate(legitimate) if data])
        else:
            MK_state = torch.tensor(np.array(state[0]), dtype=torch.float).to(self.device)
            TEC_state = torch.tensor(np.array(state[1]), dtype=torch.float).to(self.device)
            MK_Q_value = self.MK_q_net(MK_state)
            TEC_Q_value = self.TEC_q_net(TEC_state)
            illegal_mask = torch.tensor([float('-inf') if not leg else 0 for leg in legitimate], device=self.device)
            MK_Q_value = MK_Q_value + illegal_mask
            TEC_Q_value = TEC_Q_value + illegal_mask
            MK_Q_value = F.softmax(MK_Q_value, dim=-1)
            TEC_Q_value = F.softmax(TEC_Q_value, dim=-1)
            Q_value = (self.MK_W * MK_Q_value + (1 - self.MK_W) * TEC_Q_value)
            action = Q_value.argmax().item()
        return action
