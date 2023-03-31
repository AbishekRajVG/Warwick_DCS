import numpy as np

class Bot(object):
    def __init__(self):
        self.name = "nash_bot"
        self.opponent_models = {}  # opponent models for each other bot
        self.prev_bids = []  # list of previous bids
        self.round = 0  # current round number
        self.current_painting = None  # current painting being bid on
        self.painting_values = None  # dictionary of painting values
        self.opponent_strength = 0.5  # strength of opponent modeling
        self.min_bid_ratio = 0.1  # minimum bid ratio relative to painting value

    def update_opponent_model(self, opponent_id, opponent_bid):
        if opponent_id not in self.opponent_models:
            self.opponent_models[opponent_id] = [1, 1]  # initialize to uniform distribution
        n_wins, n_losses = self.opponent_models[opponent_id]
        if opponent_bid > self.prev_bids[-1]:
            self.opponent_models[opponent_id] = [n_wins + 1, n_losses]
        else:
            self.opponent_models[opponent_id] = [n_wins, n_losses + 1]

    def get_opponent_prediction(self, opponent_id):
        if opponent_id not in self.opponent_models:
            return np.random.uniform()  # if no model, return random guess
        n_wins, n_losses = self.opponent_models[opponent_id]
        total = n_wins + n_losses
        if total == 0:
            return np.random.uniform()  # if no data, return random guess
        return n_wins / total

    def get_bid(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        # update current state
        self.round = current_round
        self.budget = my_bot_details['budget']
        self.current_painting = current_painting
        self.painting_values = dict(artists_and_values)

        # track previous bids
        if len(winner_ids) > len(self.prev_bids):
            winner_id = winner_ids[-1]
            amount_paid = amounts_paid[-1]
            if winner_id != self.name:
                self.update_opponent_model(winner_id, amount_paid)
            self.prev_bids.append(amount_paid)

        # compute Nash equilibrium strategy
        opponent_predictions = np.array([self.get_opponent_prediction(bot['bot_unique_id']) for bot in bots if bot['bot_unique_id'] != self.name])
        opponent_predictions = opponent_predictions ** self.opponent_strength  # apply strength parameter
        norm_const = 1 / (1 + np.sum(opponent_predictions))
        strategy = norm_const * (1 - opponent_predictions)

        # make bid based on Nash equilibrium
        painting_value = self.painting_values[current_painting]
        min_bid = max(self.min_bid_ratio * painting_value, 1)
        nash_bid = strategy.reshape(1,-1).dot(np.arange(min_bid, self.budget+1))
        return min(nash_bid, self.budget)
