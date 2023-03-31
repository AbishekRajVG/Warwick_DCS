import random

class Bot(object):
    # def __init__(self, name, unique_id, paintings, budget, score):
    #     self.name = name
    #     self.unique_id = unique_id
    #     self.paintings = paintings
    #     self.budget = budget
    #     self.score = score

    def __init__(self):
		self.name = "budget_based_bot" # Put your id number her. String or integer will both work
		# Add your own variables here, if you want to. 
        
    def get_budget_based_bid(self, current_round, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        # calculate the average value of paintings in this round
        values = [value for artist, value in artists_and_values if artist in current_painting]
        avg_value = sum(values) / len(values) if values else 0
        
        # calculate the average budget and score of other bots in this round
        other_bots = [bot for bot in bots if bot.unique_id != self.unique_id]
        avg_budget = sum([bot.budget for bot in other_bots]) / len(other_bots) if other_bots else starting_budget
        avg_score = sum([bot.score for bot in other_bots]) / len(other_bots) if other_bots else 0
        
        # calculate the maximum bid based on the budget and score of other bots, the average value of paintings, and the current round number
        max_bid = min(self.budget, max(1, (avg_budget - self.budget) / (len(other_bots) + 1)) * ((self.score + avg_score) / (2 * current_round)) * (avg_value + 1))
        
        # make a random bid between 1 and the maximum bid
        bid = random.randint(1, max_bid)
        
        return bid

    def get_bid(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        try:
            bid = self.get_budget_based_bid(current_round, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid)
        except:
            bid = 0
            
        if bid > self.budget:
            bid = 0
            
        return int(bid)
