import random
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from collections import namedtuple

# Define the game parameters
NUM_ROUNDS = 10
BUDGET = 100
NUM_BOTS = 3

# Define the parameters for the reinforcement learning algorithm
GAMMA = 0.99
EPSILON = 0.1
LEARNING_RATE = 0.001

# Define a named tuple to store experiences in the bot's memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    """
    Deep Q-Network for reinforcement learning
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Bot(object):
    def __init__(self):
        self.name = "mix_bot_two"
        self.n_actions = 2 # bid or pass
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.lr = LEARNING_RATE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(6, self.n_actions).to(self.device)
        self.target_net = DQN(6, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = []
        self.opponent_models = {}

    def get_state(self, current_round, amounts_paid, winner_ids, budgets):
        """
        Get the current state of the game
        """
        # Get the previous round winner's id
        previous_winner = winner_ids[-1] if winner_ids else None
        
        # Calculate the average bid from the previous round
        if amounts_paid:
            avg_bid = np.mean(amounts_paid)
        else:
            avg_bid = self.budget / 2.0
        
        # Calculate the fraction of budget spent in the previous round
        if previous_winner == self.name:
            spent_frac = 1.0
        elif previous_winner is not None:

            spent_frac = amounts_paid[-1] / budgets[previous_winner]
        else:
            spent_frac = 0.0
        
        # Calculate the opponent strategies
        opponent_strategies = self.get_opponent_strategies(current_round, list(self.opponent_models.keys()), amounts_paid, winner_ids, budgets)
        
        # Return the state as a numpy array
        state = np.array([avg_bid / self.budget, spent_frac, opponent_strategies[self.name], 
                          opponent_strategies[(self.name + 1) % NUM_BOTS], opponent_strategies[(self.name + 2) % NUM_BOTS], 
                          current_round / NUM_ROUNDS])
        return state

    def get_opponent_strategies(self, current_round, bots, winner_ids, amounts_paid, budgets):
        """
        Estimate the strategies of the opponents
        """
        opponent_strategies = {}
        for bot in bots:
            bot_id = bot['bot_unique_id']
            bot_name = bot['bot_name']
            if self.name == bot_name:
                continue
            opponent_history = []
            for i, winner_id in enumerate(winner_ids):
                if winner_id == bot_id:
                    opponent_history.append(amounts_paid[i])
                if not opponent_history:
                    opponent_strategies[bot] = 0.5
                else:
                    opponent_history = np.array(opponent_history)
                    opponent_mean = np.mean(opponent_history)
                    opponent_std = np.std(opponent_history)
                    opponent_strategies[bot_id] = opponent_mean / self.budget

                if bot_id not in self.opponent_models:
                    self.opponent_models[bot_id] = DQN(6, self.n_actions).to(self.device)
                
                # Convert the state and action to a tensor
                state_tensor = torch.FloatTensor(self.get_state(current_round, amounts_paid, winner_ids, budgets)).to(self.device)
                action_tensor = torch.LongTensor([int(opponent_strategies[bot] > 0.5)]).to(self.device)
                
                # Update the opponent model with the new experience
                experience = Experience(state_tensor, action_tensor, state_tensor, torch.tensor([0.0]).to(self.device))
                self.opponent_models[bot].memory.push(experience)
                self.optimize_opponent_model(bot)

        return opponent_strategies

    def optimize_opponent_model(self, bot):
        """
        Optimize the opponent model using experience replay
        """
        # Sample experiences from the memory
        experiences = self.opponent_models[bot].memory.sample()
        batch = Experience(*zip(*experiences))

        # Convert the batch of experiences to tensors
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Calculate the predicted Q values for the current state and action
        q_values = self.opponent_models[bot](state_batch).gather(1, action_batch.unsqueeze(1))

        # Calculate the target Q values for the next state
        target_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Calculate the expected Q values
        expected_q_values = reward_batch + (self.gamma * target_q_values)

        # Calculate the loss
        loss = nn.functional.mse_loss(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.opponent_models[bot].optimizer.zero_grad()
        loss.backward()
        self.opponent_models[bot].optimizer.step()

    def choose_action(self, state):
        """
        Choose an action using the epsilon-greedy policy
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    # def get_bid_amount(self, current_round, bids, winners, budgets, bots):
    def get_bid_amount(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):

        self.budget = my_bot_details['budget']

        ##build a budget dictionary
        budgets={}

        for bot in bots:
            bot_id = bot['bot_unique_id']
            bot_remaining_budget = bot['budget']
            budgets[bot_id] = bot_remaining_budget

        """
        Get the bid amount for the current round
        """
        # Estimate the opponent strategies
        opponent_strategies = self.get_opponent_strategies(current_round, bots, amounts_paid, winner_ids, budgets)

        # Choose an action using the current state and the opponent strategies
        state = self.get_state(current_round, amounts_paid, winner_ids, budgets)
        action = self.choose_action(state)

        # Update the epsilon value
        self.epsilon = max(0.01, self.epsilon * 0.999)

        # Calculate the bid amount based on the chosen action and the opponent strategies
        if action == 0:
            bid_amount = self.budget * opponent_strategies[self.name]
        else:
            bid_amount = 0.0

        return bid_amount
    
    def get_bid(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        
        """Strategy for value type games using rule-based approach. 

		Parameters:
		current_round(int): 			The current round of the auction game
		bots(dict): 					A dictionary holding the details of all of the bots in the auction
										Includes what paintings the other bots have won and their remaining budgets
		artists_and_values(dict):		A dictionary of the artist names and the painting value to the score (for value games)
		round_limit(int):				Total number of rounds in the game
		starting_budget(int):			How much budget each bot started with
		painting_order(list str):		A list of the full painting order
		my_bot_details(dict):			Your bot details. Same as in the bots dict, but just your bot. 
										Includes your current paintings, current score and current budget
		current_painting(str):			The artist of the current painting that is being bid on
		winner_ids(list str):			List of which bots have won the rounds so far, in round order
		amounts_paid(list int):			List of amounts paid for paintings in the rounds played so far 

		Returns:
		int: Your bid. Return your bid for this round. 
		"""

        bid_amount = self.get_bid_amount(current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid)

        #if we don't already have a bid amount by this point, then don't bid.
        return 0











class Bot(object):

	def __init__(self):
		self.name = "mix_bot_two" # Put your id number here. String or integer will both work


	def get_bid(self, current_round, bots, artists_and_values, round_limit,
			starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
		
		"""Strategy for value type games using rule-based approach. 

		Parameters:
		current_round(int): 			The current round of the auction game
		bots(dict): 					A dictionary holding the details of all of the bots in the auction
										Includes what paintings the other bots have won and their remaining budgets
		artists_and_values(dict):		A dictionary of the artist names and the painting value to the score (for value games)
		round_limit(int):				Total number of rounds in the game
		starting_budget(int):			How much budget each bot started with
		painting_order(list str):		A list of the full painting order
		my_bot_details(dict):			Your bot details. Same as in the bots dict, but just your bot. 
										Includes your current paintings, current score and current budget
		current_painting(str):			The artist of the current painting that is being bid on
		winner_ids(list str):			List of which bots have won the rounds so far, in round order
		amounts_paid(list int):			List of amounts paid for paintings in the rounds played so far 

		Returns:
		int: Your bid. Return your bid for this round. 
		"""

		# # update our known paintings set
		# self.known_paintings.add(current_painting)

		# # if we haven't seen this painting before, bid 0
		# if len(self.known_paintings) == 1:
		# 	return 0

		# # check if the current painting is one we consider high value
		# if current_painting in self.high_value_paintings:
		# 	# bid up to half of our budget
		# 	my_budget = my_bot_details["budget"]
		# 	return min(my_budget//2, artists_and_values[current_painting])

		# # otherwise, we want to consider adding the painting to our high value set
		# # if it has a higher value than our current highest painting
		# highest_value_painting = max(self.high_value_paintings, key=lambda p: artists_and_values[p]) if self.high_value_paintings else None
		# if not highest_value_painting or artists_and_values[current_painting] > artists_and_values[highest_value_painting]:
		# 	# if we can afford it, bid up to the value of the painting
		# 	my_budget = my_bot_details["budget"]
		# 	if artists_and_values[current_painting] <= my_budget:
		# 		return artists_and_values[current_painting]




		# if we've made it this far, we don't want to bid on this painting
		return 0

