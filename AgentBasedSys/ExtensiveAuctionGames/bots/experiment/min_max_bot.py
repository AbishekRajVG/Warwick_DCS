import random

class Bot(object):
    def __init__(self):
        self.name = "min_max_bot"
        self.remaining_budget = 0
        self.score = 0
        
    def get_bid(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        num_bots = len(bots)
        total_budget = sum(bot['budget'] for bot in bots)
        avg_budget = total_budget / num_bots
        other_bots_budgets = [bot['budget'] for bot in bots if bot['bot_unique_id'] != self.name]
        avg_other_budget = sum(other_bots_budgets) / len(other_bots_budgets)
        
        if self.remaining_budget < avg_other_budget:
            return 0
        
        max_bid = min(self.remaining_budget, artists_and_values[current_painting][self.name])
        if max_bid == 0:
            return 0
        
        return max_bid
        
    def get_other_bots_info(self, bots):
        for bot in bots:
            if bot['bot_unique_id'] != self.name:
                return bot['budget'], bot['score']
        return None, None
