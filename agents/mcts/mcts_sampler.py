import os
import sys

# Get the absolute path to the directory containing pyhanabi.py
pyhanabi_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Append that directory to sys.path
sys.path.insert(0, pyhanabi_dir)

import random

import rl_env
from pyhanabi import HanabiCard


class MCTS_Sampler(object):
    def __init__(self) -> None:
        self.deck = HanabiDeck()

    def sample_hand(self, player, player_hands, discard_pile, fireworks, card_knowledge=None, additional_cards=[]):

        hand_size = len(player_hands[player])
        sampled_hand = []

        self.deck.reset_deck()
        self.deck.remove_by_cards(discard_pile)
        self.deck.remove_by_hands(player, player_hands, card_index=-1)
        self.deck.remove_by_firework(fireworks)
        self.deck.remove_by_cards(additional_cards)

        while len(sampled_hand) < hand_size:
            sampled_hand = []

            for card_idx in range(hand_size):
                deck_card = HanabiDeck(self.deck.card_count.copy(), self.deck.total_count)

                deck_card.remove_by_own_hand(player, player_hands, card_idx)

                # TODO : remove_by_knowledge makes segementation fault when trying my own tests
                if card_knowledge:
                    deck_card.remove_by_knowledge(card_knowledge[card_idx])

                sampled_cards = deck_card.get_deck()

                if sampled_cards:
                    sampled_card = random.choice(sampled_cards)
                    sampled_hand.append(sampled_card)
                else:
                    print("Error: No cards left in the deck to sample. Restarting the sampling process.")
                    break
        
        return sampled_hand
    
    def sample_card(self, player, card_index, player_hands, discard_pile, fireworks, card_knowledge=None, additional_cards=[]):

        self.deck.reset_deck()
        self.deck.remove_by_cards(discard_pile)
        self.deck.remove_by_hands(player, player_hands, card_index)
        self.deck.remove_by_firework(fireworks)
        self.deck.remove_by_cards(additional_cards)

        # TODO : remove_by_knowledge makes segementation fault when trying my own tests
        if card_knowledge:
            self.deck.remove_by_knowledge(card_knowledge[card_index])

        sampled_cards = self.deck.get_deck()

        if sampled_cards:
            return random.choice(sampled_cards)
        
        return None

class HanabiDeck(object):
    def __init__(self, card_count=None, total_count=None) -> None:
        self.num_colors = 5
        self.num_ranks = 5

        self.num_dict = {0: 3, 1: 2, 2: 2, 3: 2, 4: 1}

        if card_count and total_count:
            self.card_count = card_count
            self.total_count = total_count
        else:
            self.card_count = []
            self.total_count = 0
            self.reset_deck()

    def get_deck(self):
        cards = []
        append_card = cards.append # Local function binding
        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                card_index = self.card_to_index(color, rank)
                for _ in range(self.card_count[card_index]):
                    append_card(HanabiCard(color, rank))
        return cards
    
    def remove_by_knowledge(self, card_knowledge):
        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                plausible_color = card_knowledge.color_plausible(color)
                plausible_rank = card_knowledge.rank_plausible(rank)
                if not (plausible_color and plausible_rank):
                    self.remove_all_cards(color, rank)
    
    def remove_by_cards(self, cards):
        remove_card = self.remove_card  # Local function binding
        for card in cards:
            remove_card(card.color(), card.rank())

    def remove_by_hands(self, player, hands, card_index=-1):
        remove_card = self.remove_card  # Local function binding
        len_hands = len(hands)
        for other_player in range(len_hands):
            if other_player == player and card_index == -1:
                continue

            for idx, card in enumerate(hands[other_player]):
                if other_player == player and idx == card_index:
                    continue
                remove_card(card.color(), card.rank())

    def remove_by_own_hand(self, player, hands, card_index):
        remove_card = self.remove_card  # Local function binding
        for idx, card in enumerate(hands[player]):
            if idx == card_index:
                continue
            remove_card(card.color(), card.rank())
    
    def remove_by_firework(self, fireworks):
        for color in range(self.num_colors):
            for rank in range(fireworks[color]):
                self.remove_card(color, rank)

    def reset_deck(self):
        self.card_count = [self.num_dict[r] for _ in range(self.num_colors) for r in range(self.num_ranks)]
        self.total_count = sum(self.card_count)

    def remove_card(self, color, rank):
        card_idx = self.card_to_index(color, rank)

        if self.card_count[card_idx] == 0:
            print(f"Error: Trying to remove card {color}-{rank} but it is not in the deck.")
            return 
        
        self.card_count[card_idx] -= 1
        self.total_count -= 1

    def remove_all_cards(self, color, rank):
        card_idx = self.card_to_index(color, rank)
        self.total_count -= self.card_count[card_idx]
        self.card_count[card_idx] = 0

    def card_to_index(self, color, rank):
        return color * self.num_ranks + rank
    
    def is_empty(self):
        return self.total_count == 0
    
    def __str__(self) -> str:
        output = []
        output.append(f"\n{'=' * 30}DECK{'=' * 30}\n")
        color_names = ['R', 'Y', 'G', 'W', 'B']

        header = "          " + "     ".join(f"Rank {rank+1}" for rank in range(self.num_ranks))
        output.append(header)

        max_counts = [self.num_dict[rank] for rank in range(self.num_ranks)]
        max_counts_line = "Max Qty:  " + "".join(f"   {count}       " for count in max_counts)
        output.append(max_counts_line)

        output.append("          " + "-" * (10 * self.num_ranks))

        for color in range(self.num_colors):
            color_char = color_names[color]
            row = f"{color_char}       |"
            for rank in range(self.num_ranks):
                index = self.card_to_index(color, rank)
                count = self.card_count[index]
                row += f"   {count:2d}      "
            output.append(row)

        output.append(f"\nTotal cards remaining in deck: {self.total_count}")
        output.append(f"{'=' * 64}\n")
        return '\n'.join(output)
    

def test_simulation():
    env = rl_env.make('Hanabi-Full', num_players=2)
    env.reset()
    mcts_sampler = MCTS_Sampler()
    done = False

    while not done:
        for agent_id in range(2):  # Assume 2 players
            observation = env.state.observation(agent_id)
            legal_moves = env.state.legal_moves()
            random_move = random.choice(legal_moves)  # Choose a random move

            #state, reward, done, _ = env.step({'action_type': 'REVEAL_COLOR', 'color': 'B', 'target_offset': 1})

            #print(env.state.observation(0).card_knowledge()[0][0].__str__())
            #print("\n \n \n")
            #print(env.state.observation(1).card_knowledge()[0][0].__str__())
            #print("\n \n \n")

            print(env.state.player_hands())

            # Simulate sampling a card
            sample_card = mcts_sampler.sample_card(agent_id, 0, env.state.player_hands(),
                                                    env.state.discard_pile(), env.state.fireworks(), 
                                                    None, #env.state.observation(agent_id).card_knowledge()[0],
                                                    [])
            
            sample_hand = mcts_sampler.sample_hand(agent_id, env.state.player_hands(),
                                                    env.state.discard_pile(), env.state.fireworks(),
                                                    None, [])
            
            print(f"\nPlayer {agent_id} sampled hand: {sample_hand}")
            
            print(f"\nPlayer {agent_id} sampled card: {sample_card}")
            
            #print(mcts_sampler.deck)  # Print the deck state after sampling

            # Apply the move in the environment
            state, reward, done, _ = env.step(random_move)

            if done:
                break

# Run the test simulation
if __name__ == "__main__":
    test_simulation()