import numpy as np
import random

class Node:
    def __init__(self, infoset, num_actions):
        self.infoset_id = infoset
        self.regret_sum = np.zeros(num_actions)
        self.strategy = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.num_actions = num_actions
        self.visited_count = 0
        self.util_sum = 0

    def get_strategy(self):
        normalizing_sum = 0
        for a in range(self.num_actions):
            if self.regret_sum[a] > 0:
                self.strategy[a] = self.regret_sum[a]
            else:
                self.strategy[a] = 0
            normalizing_sum += self.strategy[a]

        for a in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / self.num_actions

        return self.strategy

    def get_average_strategy(self):
        avg_strategy = np.zeros(self.num_actions)
        normalizing_sum = 0

        for a in range(self.num_actions):
            normalizing_sum += self.strategy_sum[a]
        for a in range(self.num_actions):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self.num_actions

        return avg_strategy

    def pretty_print(self):
        strat_str = list(
            map(lambda s: "{0:.4f}".format(s), self.get_average_strategy())
        )
        print(
            "history:{0}, avgStrat:{1}, count:{2}, util_sum:{3:.0f}, util:{4:.4f}".format(
                self.infoset_id,
                strat_str,
                self.visited_count,
                self.util_sum,
                self.util_sum / self.visited_count if self.visited_count > 0 else 0,
            )
        )


class KuhnCFR:
    def __init__(self, iterations, decksize, num_players=2, num_bets=2, num_actions=2):
        self.num_players = num_players
        self.num_bets = num_bets
        self.iterations = iterations
        self.decksize = decksize
        self.num_actions = num_actions
        self.cards = np.arange(decksize)
        self.nodes = {}

    def cfr_iterations_external(self):
        util = np.zeros(self.num_players)
        for t in range(1, self.iterations + 1):
            for i in range(self.num_players):
                random.shuffle(self.cards)
                util[i] += self.external_cfr(self.cards[:self.num_players], [], 2, 0, i, t)
        print("Average game value: {}".format(util[0] / self.iterations))
        for i in sorted(self.nodes):
            self.nodes[i].pretty_print()

    def external_cfr(self, cards, history, pot, nodes_touched, traversing_player, t):
        plays = len(history)
        acting_player = plays % self.num_players
        opponent_player = 1 - acting_player

        if plays >= 2:
            if history[-1] == 0 and history[-2] == 1:  # bet fold
                return 1 if acting_player == traversing_player else -1
            if (history[-1] == 0 and history[-2] == 0) or (
                history[-1] == 1 and history[-2] == 1
            ):  # check check or bet call, go to showdown
                if cards[acting_player] > cards[opponent_player]:
                    return pot / 2 if acting_player == traversing_player else -pot / 2
                else:
                    return -pot / 2 if acting_player == traversing_player else pot / 2

        infoset = str(cards[acting_player]) + str(history)
        if infoset not in self.nodes:
            self.nodes[infoset] = Node(infoset, self.num_actions)

        nodes_touched += 1

        if acting_player == traversing_player:
            util = np.zeros(self.num_actions)
            node_util = 0
            strategy = self.nodes[infoset].get_strategy()
            for a in range(self.num_actions):
                next_history = history + [a]
                util[a] = self.external_cfr(cards, next_history, pot + a, nodes_touched, traversing_player, t)
                node_util += strategy[a] * util[a]

            for a in range(self.num_actions):
                regret = util[a] - node_util
                self.nodes[infoset].regret_sum[a] += regret
            self.nodes[infoset].util_sum += node_util
            self.nodes[infoset].visited_count += 1
            return node_util

        else:
            strategy = self.nodes[infoset].get_strategy()
            if random.random() < strategy[0]:
                next_history = history + [0]
            else:
                next_history = history + [1]
                pot += 1
            util = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
            for a in range(self.num_actions):
                self.nodes[infoset].strategy_sum[a] += strategy[a]
            return util


if __name__ == "__main__":
    k = KuhnCFR(10000, 3)
    k.cfr_iterations_external()
