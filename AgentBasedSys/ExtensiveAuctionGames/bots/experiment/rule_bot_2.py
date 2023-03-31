import random

class Bot(object):
    def __init__(self):
        self.name = "rule_bot_2"  # Replace with your student ID number
        self.bidding_rules = [
            {"condition": lambda round, painting: round == 1 and painting == "Picasso",
             "bid": lambda budget: budget // 2},
            {"condition": lambda round, painting: painting in ["Da Vinci", "Van Gogh"],
             "bid": lambda budget: budget // 3},
            {"condition": lambda round, painting: round >= 2 and painting != "Picasso",
             "bid": lambda budget: budget // 4},
            {"condition": lambda round, painting: True,
             "bid": lambda budget: random.randint(0, budget)}
        ]

    def get_bid(self, current_round, bots, artists_and_values, round_limit,
                starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        for rule in self.bidding_rules:
            if rule["condition"](current_round, current_painting):
                return min(my_bot_details["budget"], rule["bid"](my_bot_details["budget"]))
        return 0
