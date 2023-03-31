class Bot(object):

	def __init__(self):
		self.name = "rule_based_bot" # Put your id number here. String or integer will both work
		self.known_paintings = set() # keep track of the paintings we know about
		self.high_value_paintings = set() # keep track of the paintings we consider high value

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

		# update our known paintings set
		self.known_paintings.add(current_painting)

		# if we haven't seen this painting before, bid 0
		if len(self.known_paintings) == 1:
			return 0

		# check if the current painting is one we consider high value
		if current_painting in self.high_value_paintings:
			# bid up to half of our budget
			my_budget = my_bot_details["budget"]
			return min(my_budget//2, artists_and_values[current_painting])

		# otherwise, we want to consider adding the painting to our high value set
		# if it has a higher value than our current highest painting
		highest_value_painting = max(self.high_value_paintings, key=lambda p: artists_and_values[p]) if self.high_value_paintings else None
		if not highest_value_painting or artists_and_values[current_painting] > artists_and_values[highest_value_painting]:
			# if we can afford it, bid up to the value of the painting
			my_budget = my_bot_details["budget"]
			if artists_and_values[current_painting] <= my_budget:
				return artists_and_values[current_painting]

		# if we've made it this far, we don't want to bid on this painting
		return 0
