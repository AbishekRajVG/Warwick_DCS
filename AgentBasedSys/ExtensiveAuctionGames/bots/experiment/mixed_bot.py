## opponent modeling and reinforcement learning

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Bot(object):
    def __init__(self):
        self.name = "mixed_bot"
        # self.budget = budget
        self.n_actions = 3
        self.epsilon = 0.1
        self.gamma = 0.99
        self.lr = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(5, self.n_actions).to(self.device)
        self.target_net = DQN(5, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = []

    def get_state(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):

        self.budget = my_bot_details['budget']
        # Get the previous round winner's id
        previous_winner_id = winner_ids[-1] if winner_ids else None
        
        # Calculate the average bid from the previous round
        if amounts_paid:
            avg_bid = np.mean(amounts_paid)
        else:
            avg_bid = self.budget / 2.0
        
        # Calculate the fraction of budget spent in the previous round
        if previous_winner_id == self.name:
            spent_frac = 1.0
        elif amounts_paid:
            spent_frac = amounts_paid[-1] / self.budget
        else:
            spent_frac = 0.0
        
        # Calculate the fraction of budget remaining for the current round
        budget_frac = my_bot_details['budget'] / self.budget
        
        # Calculate the opponent strategies
        opponent_strategies = self.get_opponent_strategies(current_round, bots, painting_order, winner_ids, amounts_paid)
        print("openent strategies")
        print(opponent_strategies)
        
        state = [
            current_round / round_limit,
            avg_bid / self.budget,
            spent_frac,
            budget_frac,
            opponent_strategies[self.name]
        ]
        return np.array(state, dtype=np.float32)

    def get_opponent_strategies(self, current_round, bots, painting_order, winner_ids, amounts_paid):
        opponent_strategies = {}
        for bot in bots:
            bot_id = bot['bot_unique_id']
            if bot['bot_name'] == self.name:
                continue
            opponent_history = []
            for i, winner_id in enumerate(winner_ids):
                if winner_id == bot_id:
                    opponent_history.append(amounts_paid[i])
            if not opponent_history:
                
                opponent_strategies[bot_id] = 0.5
            else:
                opponent_history = np.array(opponent_history)
                opponent_mean = np.mean(opponent_history)
                opponent_std = np.std(opponent_history)
                opponent_strategies[bot] = max(0.05, min(0.95, (opponent_mean + opponent_std) / self.budget))

        return opponent_strategies
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        else:
            with torch.no_grad():
                state = torch.tensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()
                return action

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state = zip(*random.sample(self.memory, batch_size))
        
        state = torch.tensor(state).to(self.device)
        action = torch.tensor(action).unsqueeze(1).to(self.device)
        reward = torch.tensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.tensor(next_state).to(self.device)
        
        current_q = self.policy_net(state).gather(1, action)
        next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
        target_q = reward + self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, n_episodes, batch_size, env):
        for episode in range(n_episodes):
            state = env.reset()
            for t in range(env.round_limit):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action, self)
                self.remember(state, action, reward, next_state)
                state = next_state
                self.replay(batch_size)
                if done:
                    self.update_target_net()
                    break

    # def get_bid(self, state):
    #     q_values = self.policy_net(torch.tensor(state).unsqueeze(0).to(self.device))
    #     return q_values.max().item() * self.budget

    def get_bid(self, current_round, bots, artists_and_values, round_limit,
		starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):

        state = self.get_state(current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid)

        q_values = self.policy_net(torch.tensor(state).unsqueeze(0).to(self.device))

        return q_values.max().item() * self.budget


