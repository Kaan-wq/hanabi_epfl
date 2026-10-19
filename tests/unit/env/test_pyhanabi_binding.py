import re
from pathlib import Path
from typing import Any

import pytest

from hanabi_search.env import HanabiGame, _pyhanabi
from hanabi_search.exceptions import EngineLoadError
from tests.support import REPO_ROOT


def _declarations(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return " ".join(text.split())


def test_cdef_matches_c_api_header() -> None:
    header = (REPO_ROOT / "engine" / "capi" / "pyhanabi.h").read_text()
    block = re.search(r'extern "C" \{(.*)\} /\* extern "C" \*/', header, flags=re.DOTALL)
    assert block is not None
    assert _declarations(_pyhanabi.CDEF) == _declarations(block.group(1))


def test_library_is_loaded() -> None:
    assert _pyhanabi.lib.NumPlayers is not None
    assert _pyhanabi.LIBRARY_PATH.is_file()


def test_missing_library_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_pyhanabi.LIB_ENV_VAR, str(tmp_path / "missing.dylib"))
    with pytest.raises(EngineLoadError, match="not found"):
        _pyhanabi._load_library()


def test_unloadable_library_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bogus = tmp_path / "libpyhanabi.dylib"
    bogus.write_text("not a library")
    monkeypatch.setenv(_pyhanabi.LIB_ENV_VAR, str(bogus))
    with pytest.raises(EngineLoadError, match="could not load"):
        _pyhanabi._load_library()


def test_take_string_decodes_engine_string() -> None:
    game = HanabiGame({"players": 3, "seed": 1})
    assert _pyhanabi.take_string(_pyhanabi.lib.GameParamString(game.c_game)).count("players=3") == 1


def test_card_values_reads_struct() -> None:
    c_card: Any = _pyhanabi.ffi.new("pyhanabi_card_t*")  # cffi struct fields are untyped.
    c_card.color, c_card.rank = 3, 1
    assert _pyhanabi.card_values(c_card) == (3, 1)
