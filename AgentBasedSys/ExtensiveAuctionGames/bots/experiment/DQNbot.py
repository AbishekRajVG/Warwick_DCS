import numpy as np
import tensorflow as tf
import random

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = []
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.model.predict(np.array([state]))[0]
            return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        samples = np.array(random.sample(self.memory, self.batch_size))
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])
        targets = self.model.predict(states)
        next_q_values = self.model.predict(next_states).max(axis=1)
        targets[np.arange(self.batch_size), actions] = rewards + (1 - dones) * self.discount_factor * next_q_values
        self.model.fit(states, targets, epochs=1, verbose=0)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Bot(object):

    def __init__(self):
        self.name = "DQNbot" 
    
    def get_bid(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):

        

        # Get the current state
        state = self.state_encoder(current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid)

        # Instantiate a DQNAgent with the appropriate state and action sizes
        state_size = len(state)
        action_size = my_bot_details['budget']  # Bid amounts from 0 to budget inclusive
        agent = DQNAgent(state_size, action_size)

        # Get the bid amount using the DQNAgent
        bid_amount = agent.act(state)

        # Make sure the bid amount is within the budget limit
        bid_amount = min(bid_amount, my_bot_details['budget'])

        return bid_amount
        

    def state_encoder(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        my_budget = my_bot_details['budget']
        my_score = my_bot_details['score']
        highest_bid = max(amounts_paid) if len(amounts_paid) > 0 else 1
        # highest_bid = max(bots.values(), key=lambda bot: bot['bid'])['bid'] if bots else 0
        remaining_rounds = round_limit - current_round

        max_possible_score = my_score + \
            artists_and_values[current_painting] + \
            sum([artists_and_values[artist] for artist in painting_order[current_round:]])

        artist_score = artists_and_values[current_painting]
        is_last_painting = current_painting == painting_order[-1]

        ## a successful bids dictionary
        bid_history={}
        for i, winner_id in enumerate(winner_ids):
            if winner_id not in bid_history:
                bid_history[winner_id] = amounts_paid[i]
            else:
                if bid_history[winner_id] < amounts_paid[i]:
                    bid_history[winner_id] = amounts_paid[i]
        
        for bot in bots:
            bot_id = bot['bot_unique_id']
            if bot_id not in bid_history:
                bid_history[bot_id] = 0

        budget_ratios = [bot['budget'] / starting_budget \
                         for bot in bots if bot['bot_unique_id'] != my_bot_details['bot_unique_id']]
        winner_ratios = [winner_ids.count(bot['bot_unique_id']) / (current_round+1) \
                         for bot in bots if bot['bot_unique_id'] != my_bot_details['bot_unique_id']]
        bid_ratios = [(bid_history[bot['bot_unique_id']] / starting_budget) / (highest_bid / starting_budget) \
                      for bot in bots if bot['bot_unique_id'] != my_bot_details['bot_unique_id']]
        values = [artists_and_values[artist] for artist in painting_order[current_round:]]
        
        state = [current_round, my_budget, my_score, highest_bid, remaining_rounds, max_possible_score,
                artist_score, is_last_painting] + budget_ratios + winner_ratios + bid_ratios + values

        return state
    


