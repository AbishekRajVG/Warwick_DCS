import random

class Bot(object):

    def __init__(self):
        self.name = "MDPbot"
        # self.budget = budget
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        max_q = max(self.q_table[next_state].values()) if next_state in self.q_table else 0
        td_target = reward + self.discount_factor * max_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def get_best_action(self, state):
        if random.random() < self.epsilon:
            # Explore
            return random.choice(['bid', 'pass'])
        else:
            # Exploit
            q_values = self.q_table.get(state, {})
            if q_values:
                best_action = max(q_values, key=q_values.get)
                return best_action
            else:
                return random.choice(['bid', 'pass'])

    def get_bid(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        state = self.get_state(current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid)

        # print("current bot details")
        # print(my_bot_details)

        # print("winner Ids: ")
        # print(winner_ids)

        # print("amount_bids: ")
        # print(amounts_paid)

        # print( "painting order: ")
        # print(painting_order)

        # print( "round limit")
        # print(round_limit)


        action = self.get_best_action(state)
        if action == 'bid':
            max_bid = min(my_bot_details["budget"], artists_and_values[current_painting])
            return max(1, random.randint(1, max_bid))
        else:
            return 0

    def get_state(self, current_round, bots, artists_and_values, round_limit, starting_budget, painting_order, my_bot_details, current_painting, winner_ids, amounts_paid):
        # Encode state as a string for use as a dictionary key
        state = f"{current_round},{my_bot_details['budget']},{current_painting}"
        # print(bots)
        for bot in bots:
            if bot['bot_name'] != self.name:
                state += f",{bot['budget']},{bot['score']}"
        # for bot_name, bot_details in bots.items():
        #     if bot_name != self.name:
        #         state += f",{bot_details['budget']},{bot_details['score']}"
        return state
