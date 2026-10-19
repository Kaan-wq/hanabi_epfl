import pytest

from hanabi_search.env import (
    HanabiCard,
    HanabiMove,
    HanabiMoveType,
    color_char_to_idx,
    color_idx_to_char,
)
from hanabi_search.exceptions import SerializationError


def test_color_conversions_round_trip() -> None:
    for index in range(5):
        char = color_idx_to_char(index)
        assert char is not None
        assert color_char_to_idx(char) == index
    assert color_idx_to_char(-1) is None


def test_color_conversions_reject_invalid() -> None:
    with pytest.raises(IndexError):
        color_idx_to_char(5)
    with pytest.raises(ValueError, match="Invalid color"):
        color_char_to_idx("P")


def test_card_value_semantics() -> None:
    card = HanabiCard(1, 3)
    assert card == HanabiCard(1, 3)
    assert card != HanabiCard(1, 2)
    assert len({card, HanabiCard(1, 3)}) == 1
    assert str(card) == "Y4"
    assert card.to_dict() == {"color": "Y", "rank": 3}


def test_hidden_card() -> None:
    card = HanabiCard(-1, -1)
    assert not card.valid()
    assert str(card) == "XX"
    assert card.to_dict() == {"color": None, "rank": -1}


@pytest.mark.parametrize(
    ("move", "expected"),
    [
        (HanabiMove.get_play_move(2), {"action_type": "PLAY", "card_index": 2}),
        (HanabiMove.get_discard_move(0), {"action_type": "DISCARD", "card_index": 0}),
        (
            HanabiMove.get_reveal_color_move(1, 4),
            {"action_type": "REVEAL_COLOR", "target_offset": 1, "color": "B"},
        ),
        (
            HanabiMove.get_reveal_rank_move(2, 0),
            {"action_type": "REVEAL_RANK", "target_offset": 2, "rank": 0},
        ),
        (HanabiMove.get_return_move(1, 3), {"action_type": "RETURN", "card_index": 1}),
        (
            HanabiMove.get_deal_specific_move(0, 1, 2, 3),
            {"action_type": "DEAL_SPECIFIC", "color": "G", "rank": 3},
        ),
    ],
)
def test_move_to_dict(move: HanabiMove, expected: dict[str, object]) -> None:
    assert move.to_dict() == expected


def test_move_json_round_trip_and_hash() -> None:
    move = HanabiMove.get_reveal_rank_move(1, 2)
    restored = HanabiMove.from_json(move.to_json())
    assert restored == move
    assert hash(restored) == hash(move)
    assert restored.type() is HanabiMoveType.REVEAL_RANK
    assert str(restored) == str(move)


def test_move_from_bad_json_raises() -> None:
    with pytest.raises(SerializationError):
        HanabiMove.from_json("{not json")
