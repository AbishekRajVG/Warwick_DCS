
# Import the auctioneer, to run your auctions
from auctioneer import Auctioneer

# Import some bots to play with
# We have given you some basic bots to start with in the bots folder
# You can also add your own bots to test against
from bots import flat_bot_10
from bots import random_bot
from bots import u1234321
from bots import u2238887
# from bots import logic_based_bot
# from bots import Nash_bot
# from bots import rule_based_bot, rule_bot_2, vickery_bot, min_max_bot
# from bots.experiment import MDPbot, Nashbot
from bots.experiment import logic_based_bot, rule_based_bot, rule_bot_2, min_max_bot, MDPbot, Nashbot, NashBot2, mixed_bot, DQNbot, vickery_bot

def run_basic_auction():
	"""
	An example function that runs a basic auction with 3 bots
	"""
	# Setup a room of bots to play against each other, imported above from the bots folder
	room = [u1234321, flat_bot_10, random_bot]

	# Setup the auction
	my_auction = Auctioneer(room=room)
	# Play the auction
	my_auction.run_auction()

def run_auction_with_my_bot():
	"""
	An example function that runs a basic auction with 3 bots
	"""
	# Setup a room of bots to play against eac[h other, imported above from the bots folder
	# room = [u1234321, flat_bot_10, random_bot, u2238887, logic_based_bot, Nash_bot, rule_based_bot, rule_bot_2]
	# room = [logic_based_bot, rule_based_bot, rule_bot_2, min_max_bot, MDPbot, DQNbot]
	# room = [DQNbot, u1234321]
	room = [u2238887, random_bot, vickery_bot]


	# Setup the auction
	my_auction = Auctioneer(room=room)
	# Play the auction
	my_auction.run_auction()
	

def run_lots_of_auctions():
	"""
	An example if you want to run alot of auctions at once
	"""
	# A large room with a few bots of the same type
	room = [random_bot, random_bot, random_bot, u1234321]
	
	win_count = 0
	run_count = 50
	for i in range(run_count):
		# Setup the auction
		# slowdown = 0 makes it fast
		my_auction = Auctioneer(room=room, slowdown=0)
		# run_auction() returns a list of winners, sometimes there are more than one winner if there is a tie
		winners = my_auction.run_auction()
		# Check if the bot's name, "my_bot", was a winner 
		if "1234321" in winners:
			win_count +=1
	print("My bot won {} of {} games".format(win_count, run_count))	


if __name__=="__main__":
	# run_basic_auction()
	run_auction_with_my_bot()