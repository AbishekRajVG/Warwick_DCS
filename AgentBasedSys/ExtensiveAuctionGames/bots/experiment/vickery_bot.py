class Bot(object):
    def __init__(self):
        self.name = "vickrey_bot"  # replace with your own ID
        # add any other variables here, if needed

    def get_bid(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        # get my current budget
        my_budget = my_bot_details["budget"]

        # get the value of the current painting
        current_painting_value = artists_and_values[current_painting]

        # get the maximum bid by the other bots
        other_bots = [bot for bot in bots if bot['bot_name'] != self.name]
        # other_bots = [bot for bot in bots.keys() if bot != self.name]

        max_other_bid = 0
        if current_round > 1: 
            for bot in other_bots:
                bot_id = bot['bot_unique_id']
                for i, winner_id in enumerate(winner_ids):
                    if winner_id == bot_id:
                        max_other_bid = max(max_other_bid, amounts_paid[i])
                    
            # if current_round > 1: 
            #     if winner_ids[current_round-1] == bot_id:
            #         max_other_bid = max(max_other_bid, amounts_paid[current_round-1])
            #     else:
            #         max_other_bid = max(max_other_bid, bot["last_bid"])

        # calculate my bid using Vickrey auction strategy
        my_bid = min(my_budget, max_other_bid)

        return my_bid
