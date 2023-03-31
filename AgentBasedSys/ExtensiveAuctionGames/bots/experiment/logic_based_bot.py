import random

class Bot(object):

	def __init__(self):
		self.name = "logic_based_bot"
		self.high_value_artists = ["Van Gogh", "Monet", "Picasso"]
		self.low_value_artists = ["Renoir", "Cezanne", "Gauguin"]
		# add any other variables you need here

	def get_bid(self, current_round, bots, artists_and_values, round_limit,
			starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):

		# if the painting is from a high value artist, bid aggressively
		if current_painting in self.high_value_artists:
			return int(my_bot_details["budget"] * 0.8)

		# if the painting is from a low value artist, bid conservatively
		if current_painting in self.low_value_artists:
			return int(my_bot_details["budget"] * 0.2)

		# if we don't have any other strategy, bid randomly
		return random.randint(0, my_bot_details["budget"])
