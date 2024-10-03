# Record a game by tracking moves and the game state observations
from pyhanabi import HanabiMoveType

num_rank = [3, 2, 2, 2, 1]

class RecordMoves(object):

    def __init__(self, players):
        self._stat_list = ['score', 'moves', 'regret', 'discard', 'play', 'information']
        self.recorded_observation = None
        self.players = players
        self.game_stats = self.default_stats()
        self.player_stats = [self.default_stats() for _ in range(self.players)]

    def reset(self, observations):
        """Reset stats and recorded observations."""
        self.recorded_observation = observations
        self.game_stats = self.default_stats()
        self.player_stats = [self.default_stats() for _ in range(self.players)]

    def default_stats(self):
        """Create default stats with zero values."""
        return {s: 0 for s in self._stat_list}

    def update(self, move, observation, action_player, elapsed_time):
        """Update game stats by passing the action taken and the new state observation."""
        self.game_stats["score"] = self._score(observation)
        self.player_stats[action_player]["score"] = self._score(observation)
        self._update_stat("moves", 1, action_player)

        # Reveal information
        if move.type() == HanabiMoveType.REVEAL_RANK or move.type() == HanabiMoveType.REVEAL_COLOR:
            self._update_stat("information", 1, action_player)

        # Discard action
        if move.type() == HanabiMoveType.DISCARD:
            self._update_stat("discard", 1, action_player)
            card = observation["discard_pile"][-1]
            if self._critical_discard(card, observation):
                regret = self._critical_card_regret(observation, self.recorded_observation)
                self._update_stat("regret", regret, action_player)

        # Play action
        if move.type() == HanabiMoveType.PLAY:
            self._update_stat("play", 1, action_player)
            if observation["life_tokens"] < self.recorded_observation["life_tokens"]:
                regret = self._critical_card_regret(observation, self.recorded_observation)
                self._update_stat("regret", regret, action_player)

        # Record the current observation for future comparisons
        self.recorded_observation = observation

    def _update_stat(self, stat, increment, action_player):
        """Update the given stat for both the game and the action player."""
        self.game_stats[stat] += increment
        self.player_stats[action_player][stat] += increment

    def _critical_card_regret(self, observation, recorded_observation):
        """Calculate regret when a critical card is discarded or misplayed."""
        max_fireworks_before = self._get_max_fireworks(recorded_observation)
        max_fireworks_after = self._get_max_fireworks(observation)
        regret = self._fireworks_score(max_fireworks_before) - self._fireworks_score(max_fireworks_after)
        return min(regret, observation["turns_to_play"])

    def _score(self, observation):
        """Calculate the score based on fireworks."""
        if observation["life_tokens"] == 0:
            return 0
        else:
            return self._fireworks_score(observation["fireworks"])

    def _fireworks_score(self, fireworks):
        """Compute the fireworks score as the sum of completed stacks."""
        return sum(v for k, v in fireworks.items())

    def _critical_discard(self, card, observation):
        """Check if the discarded card was critical for gameplay."""
        num = self._count_card(card, observation["discard_pile"])
        if num == num_rank[card["rank"]] and observation["fireworks"][card["color"]] < card["rank"] + 1:
            return True
        return False

    def _get_max_fireworks(self, observation):
        """Get the maximum possible fireworks score based on discarded cards."""
        discarded_cards = {}
        max_fireworks = {'R': 5, 'Y': 5, 'G': 5, 'W': 5, 'B': 5}
        for card in observation['discard_pile']:
            color = card['color']
            rank = card['rank']
            label = str(color) + str(rank)
            discarded_cards[label] = discarded_cards.get(label, 0) + 1

        for label, count in discarded_cards.items():
            color = label[0]
            rank = int(label[1])
            if count >= num_rank[rank] and max_fireworks[color] >= rank:
                max_fireworks[color] = rank
        return max_fireworks

    def _count_card(self, card, pile):
        """Count how many times a specific card has been discarded."""
        return sum(1 for discarded_card in pile if discarded_card["rank"] == card["rank"] and discarded_card["color"] == card["color"])

    def regret(self):
        """Return the total regret recorded in the game."""
        return self.game_stats['regret']
