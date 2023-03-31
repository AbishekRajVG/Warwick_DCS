
import random

class Bot(object):

	def __init__(self):
		self.name = "1234321" # Put your id number her. String or integer will both work
		# Add your own variables here, if you want to. 

	def get_bid(self, current_round, bots, artists_and_values, round_limit,
			starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
		
		"""Strategy for value type games. 

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
		int:Your bid. Return your bid for this round. 
		"""

		# WRITE YOUR STRATEGY HERE - MOST VALUABLE PAINTINGS WON WINS


		# Here is an example of how to get the current painting's value
		current_painting_value = artists_and_values[current_painting]
		print("The current painting's value is ", current_painting_value)

		# Here is an example of printing who won the last round
		if current_round>1:
			who_won_last_round = winner_ids[current_round-1]
			print("The last round was won by ", who_won_last_round)

		# Play around with printing out other variables in the function, 
		# to see what kind of inputs you have to work with
		my_budget = my_bot_details["budget"]
		return random.randint(0, my_budget)

