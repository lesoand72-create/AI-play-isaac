# =======================================================================
#  game_state.py — читает JSON, который пишет Lua-мод.
# =======================================================================
import json

# Тот же путь, что и OUT_PATH в main.lua.
STATE_PATH = r"C:\Users\Lesoa\isaac_rl\state.json"


def read_game_state(last_good=None):
    """Читает состояние игры из файла мода.

    Игра пишет файл ~30 раз/сек, поэтому иногда мы попадаем на момент
    записи и получаем «битый» JSON. Тогда возвращаем последнее корректное
    состояние (last_good), чтобы программа не падала.
    """
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return last_good