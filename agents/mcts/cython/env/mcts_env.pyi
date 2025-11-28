from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

from agents.mcts.cython.sampler.mcts_sampler import MCTS_Sampler
from pyhanabi import HanabiMove, HanabiMoveType, AgentObservationType, HanabiCard
from rl_env import HanabiEnv

class DetermineType(IntEnum):
    """Move types, consistent with hanabi_lib/hanabi_move.h."""
    RESTORE: int
    REPLACE: int
    NONE: int

class ScoreType(IntEnum):
    """Move types, consistent with hanabi_lib/hanabi_move.h."""
    SCORE: int
    REGRET: int
    PROGRESS: int

class MCTS_Env(HanabiEnv):
    mcts_player: int
    determine_type: int
    score_type: int
    remember_hand: Optional[List[HanabiCard]]
    sampler: MCTS_Sampler
    
    def __init__(self, config: Dict[str, Any]) -> None: ...
    
    def step(
        self, 
        action: Union[Dict[str, Any], int, HanabiMove]
    ) -> Tuple[List[Dict[str, Any]], float, bool, Dict[str, Any]]: ...
    
    def reward(self) -> float: ...
    
    def return_hand(self, player: int) -> None: ...
    
    def replace_hand(self, player: int) -> None: ...
    
    def restore_hand(
        self,
        player: int,
        remember_hand: List[HanabiCard],
        removed_card: Optional[HanabiCard] = None,
        removed_card_index: int = -1
    ) -> None: ...

def make(
    environment_name: str = "Hanabi-Full",
    num_players: int = 2,
    mcts_player: int = 0,
    determine_type: int = 0,
    score_type: int = 0,
    pyhanabi_path: Optional[str] = None,
) -> MCTS_Env: ...
