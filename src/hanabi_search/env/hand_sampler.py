"""Hidden-hand sampling for re-determinizing MCTS, done by the engine (`HandSampler`).

The sampling functions take a seed instead of using a generator of their own,
so callers decide where randomness comes from (`DeterminizedEnv` draws it from
Python's `random`, which each game reseeds).
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from hanabi_search.env._pyhanabi import CData, engine_message, ffi, lib
from hanabi_search.env.cards import HanabiCard
from hanabi_search.env.state import HanabiState
from hanabi_search.exceptions import ConfigError, EngineCallError

_SEED_LIMIT = 2**64


def _card_array(cards: Sequence[HanabiCard]) -> CData:
    """Returns `cards` as a `pyhanabi_card_t[]` (NULL when empty)."""
    if not cards:
        return ffi.NULL
    return ffi.new("pyhanabi_card_t[]", [(card.color(), card.rank()) for card in cards])


def candidate_counts(
    state: HanabiState,
    player: int,
    card_index: int,
    *,
    use_knowledge: bool = True,
    excluded: Sequence[HanabiCard] = (),
) -> npt.NDArray[np.int32]:
    """Returns how many copies of each card could be at `player`'s position `card_index`.

    Counts every card not elsewhere in the state: the deck, plus the card now at
    that position (if any), minus one copy per card in `excluded`.

    Args:
        state: The state.
        player: Absolute player id.
        card_index: Position in the hand, in [0, hand size).
        use_knowledge: Whether cards the player's hints rule out there count 0.
        excluded: Further cards known not to be available.

    Returns:
        Counts of shape (num_colors, num_ranks), dtype int32.

    Raises:
        EngineCallError: If an argument is out of range.
    """
    game = state.game
    counts = np.zeros(game.num_colors() * game.num_ranks(), dtype=np.int32)
    if not lib.pyhanabi_state_candidate_counts(
        state.c_state,
        player,
        card_index,
        use_knowledge,
        _card_array(excluded),
        len(excluded),
        ffi.from_buffer("int32_t[]", counts),
        counts.size,
    ):
        raise EngineCallError(
            engine_message(f"Cannot count candidates for player {player} at {card_index}")
        )
    return counts.reshape(game.num_colors(), game.num_ranks())


def _check_seed(seed: int) -> None:
    if not 0 <= seed < _SEED_LIMIT:
        raise ConfigError(f"Sampling seed must be in [0, 2**64), got {seed}")


def resample_hand(state: HanabiState, player: int, seed: int) -> None:
    """Replaces `player`'s hand with one sampled from what that player knows.

    The swap is not a move: the move history, and so every observation's last
    moves, the current player and the tokens are unchanged.

    Args:
        state: The state to change.
        player: Absolute player id.
        seed: Seed of the sampling, in [0, 2**64); equal seeds give equal hands.

    Raises:
        ConfigError: If `seed` is out of range.
        EngineCallError: If `player` is out of range or no consistent hand exists;
            `state` is then unchanged.
    """
    _check_seed(seed)
    if not lib.pyhanabi_state_resample_hand(state.c_state, player, seed):
        raise EngineCallError(engine_message(f"Cannot resample the hand of player {player}"))


def restore_hand(
    state: HanabiState,
    player: int,
    remembered: Sequence[HanabiCard],
    removed_card: HanabiCard | None,
    removed_index: int,
    seed: int,
) -> None:
    """Restores `player`'s hand after they acted, as closely as possible.

    Deals back `remembered` without position `removed_index`, then samples the
    card they drew. A remembered copy of `removed_card` that is no longer
    available is replaced by a sampled card.

    Args:
        state: The state, after the player's move and draw.
        player: Absolute player id.
        remembered: The player's hand before they acted.
        removed_card: The card they played or discarded; `None` after a hint.
        removed_index: Position of that card in `remembered`; -1 after a hint.
        seed: Seed of the sampling, in [0, 2**64).

    Raises:
        ConfigError: If `seed` is out of range.
        EngineCallError: If an argument is out of range or no consistent hand
            exists; `state` is then unchanged.
    """
    _check_seed(seed)
    color, rank = (-1, -1)
    if removed_card is not None:
        color, rank = removed_card.color(), removed_card.rank()
    if not lib.pyhanabi_state_restore_hand(
        state.c_state,
        player,
        _card_array(remembered),
        len(remembered),
        color,
        rank,
        removed_index,
        seed,
    ):
        raise EngineCallError(engine_message(f"Cannot restore the hand of player {player}"))
