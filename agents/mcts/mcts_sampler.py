import random

from pyhanabi import HanabiCard


class MCTS_Sampler(object):
    """Sampler for re-determinization in MCTS"""

    def __init__(self):
        self.deck = HanabiDeck()

    def sample_hand(
        self,
        player,
        original_hand_size,
        player_hands,
        discard_pile,
        fireworks,
        card_knowledge=None,
        additional_cards=[],
    ):
        sampled_hand = []

        self.deck.reset_deck()
        self.deck.remove_by_cards(discard_pile)
        self.deck.remove_by_hands(player, player_hands, card_index=-1)
        self.deck.remove_by_fireworks(fireworks)
        self.deck.remove_by_cards(additional_cards)

        while len(sampled_hand) < original_hand_size:
            sampled_hand = []

            for card_idx in range(original_hand_size):
                deck_card = HanabiDeck(self.deck.card_count.copy(), self.deck.total_count)
                deck_card.remove_by_cards(sampled_hand)
                deck_card.remove_by_own_hand(player, player_hands, card_idx)

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

    def sample_card(
        self,
        player,
        card_index,
        player_hands,
        discard_pile,
        fireworks,
        card_knowledge=None,
        additional_cards=[],
    ):
        valid_cards = self.valid_cards(
            player,
            card_index,
            player_hands,
            discard_pile,
            fireworks,
            card_knowledge,
            additional_cards,
        )
        if valid_cards:
            return random.choice(valid_cards)
        else:
            return None

    def valid_cards(
        self,
        player,
        card_index,
        player_hands,
        discard_pile,
        fireworks,
        card_knowledge=None,
        additional_cards=[],
    ):
        self.deck.reset_deck()
        self.deck.remove_by_cards(discard_pile)
        self.deck.remove_by_hands(player, player_hands, card_index)
        self.deck.remove_by_fireworks(fireworks)
        self.deck.remove_by_cards(additional_cards)

        if card_knowledge is not None:
            self.deck.remove_by_knowledge(card_knowledge[card_index])

        return self.deck.get_deck()


class HanabiDeck(object):
    """Deck of Hanabi cards for sampling hands and cards"""

    def __init__(self, card_count=None, total_count=None):
        self.num_ranks = 5
        self.num_colors = 5

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
        append_card = cards.append  # Local function binding
        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                card_index = self.card_to_index(color, rank)
                for _ in range(self.card_count[card_index]):
                    append_card(HanabiCard(color, rank))
        return cards

    def remove_by_knowledge(self, card_knowledge):
        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                if not (
                    card_knowledge.color_plausible(color)
                    and card_knowledge.rank_plausible(rank)
                ):
                    self.remove_all_card(color, rank)

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

    def remove_by_fireworks(self, fireworks):
        for color in range(self.num_colors):
            for rank in range(fireworks[color]):
                self.remove_card(color, rank)

    def reset_deck(self):
        self.card_count = [
            self.num_dict[r]
            for _ in range(self.num_colors)
            for r in range(self.num_ranks)
        ]
        self.total_count = sum(self.card_count)

    def remove_card(self, color, rank):
        card_idx = self.card_to_index(color, rank)

        if self.card_count[card_idx] == 0:
            return

        self.card_count[card_idx] -= 1
        self.total_count -= 1

    def remove_all_card(self, color, rank):
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
        color_names = ["R", "Y", "G", "W", "B"]

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

        return "\n".join(output)
