

class Bot(object):
	"""
	Bids 10 on everything
	"""
	def __init__(self):
		self.name = "flat_bot_10"

	def get_bid(self, current_round, bots, artists_and_values, round_limit,
		starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
		return 10