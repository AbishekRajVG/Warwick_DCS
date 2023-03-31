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

class DeepQLearningBot:
    def __init__(self, bot_id, budget):
        self.name = bot_id
        self.budget = budget
        self.n_actions = 3
        self.epsilon = 0.1
        self.gamma = 0.99
        self.lr = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(3, self.n_actions).to(self.device)
        self.target_net = DQN(3, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = []

    def get_state(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        state = [
            current_round / round_limit,
            my_bot_details['budget'] / starting_budget,
            artists_and_values[current_painting] / starting_budget
        ]
        return np.array(state, dtype=np.float32)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
                return action

    def update_policy(self):
        if len(self.memory) < 64:
            return
        transitions = random.sample(self.memory, 64)
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(batch.next_state).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q_values = (next_q_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_to_memory(self, state, action, reward, next_state):
        self.memory.append(Transition(state, action, reward, next_state))
        self.update_policy()

    def get_bid(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
            # Get current state
        state = self.get_state(current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid)
        
        # Determine action
        action = self.get_action(state)
        
        # Place bid
        if action == 0:
            bid_amount = 0
        elif action == 1:
            bid_amount = int(self.budget / 2)
        else:
            bid_amount = self.budget
        
        # Update memory with current state and action
        next_state = self.get_state(current_round+1, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid)
        reward = my_bot_details['score'] - bots[self.name]['score']
        self.add_to_memory(state, action, reward, next_state)
        
        return bid_amount

        
