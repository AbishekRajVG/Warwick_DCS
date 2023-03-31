import numpy as np
import scipy.optimize

class Bot(object):
    def __init__(self):
        self.name = "nash_bot_2"

    def get_bid(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        min_bid = amounts_paid[-1] + 1 if amounts_paid else artists_and_values[current_painting][1]
        n_bots = len(bots)
        self.budget = my_bot_details['budget']

        # Initialize strategy matrix
        strategy = np.zeros((n_bots, self.budget - min_bid + 1))

        # Fill in strategy matrix
        for i, bot in enumerate(bots):
            if bot == self.name:
                strategy[i, :] = 1 / strategy.shape[1]  # Uniform random bidding
            else:
                # Define utility function for opponent i
                def opponent_utility(bid):
                    opponent_bid = np.concatenate((strategy[:i], [bid], strategy[i+1:]))
                    return np.dot(opponent_bid, np.arange(min_bid, self.budget+1))

                # Find Nash equilibrium for opponent i
                initial_guess = (self.budget - min_bid + 1) * np.ones(n_bots - 1) / (n_bots - 1)
                bounds = [(min_bid, self.budget)] * (n_bots - 1)
                result = scipy.optimize.minimize(opponent_utility, initial_guess, bounds=bounds, method='SLSQP')

                # Store Nash equilibrium in strategy matrix
                strategy[i, :] = np.concatenate(([0], result.x, [0]))

        # Compute expected payoff for this bot
        expected_payoff = np.dot(strategy.dot(np.arange(min_bid, self.budget+1)), strategy[:, bots.index(self.name)])

        # Make bid at Nash equilibrium
        nash_bid = strategy[bots.index(self.name)].dot(np.arange(min_bid, self.budget+1))
        return int(nash_bid) if nash_bid < my_bot_details['budget'] else my_bot_details['budget']