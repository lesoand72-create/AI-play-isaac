# =======================================================================
#  reward_modded.py — точная reward-функция на данных Lua-мода.
# =======================================================================
from .game_state import read_game_state

R_SURVIVE_STEP = 0.01
R_DAMAGE_ENEMY = 0.30    # упало суммарное HP врагов
R_ROOM_CLEAR   = 2.00    # комната стала "clear"
R_NEW_ROOM     = 1.50    # сменился id комнаты
R_NEW_FLOOR    = 5.00    # перешли на следующий этаж
R_PICKUP_COIN  = 0.50    # стало больше монет
P_LOSE_HEART   = -3.0    # уменьшились сердца (за каждое полусердце)
P_DEATH        = -25.0   # смерть
P_BLIND_SHOT   = -0.20   # стреляет, а врагов в комнате нет


class ModdedRewardCalculator:
    """Считает награду по точным данным из state.json."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None   # предыдущее состояние игры

    def compute(self, is_shooting):
        reward = R_SURVIVE_STEP
        done = False
        info = {}

        state = read_game_state(self.prev)
        if state is None:
            return reward, done, info   # файл ещё не готов

        if self.prev is not None:
            # Суммарные сердца = красные + душевные (в полусердцах).
            prev_hp = self.prev["hearts"] + self.prev["soulHearts"]
            cur_hp  = state["hearts"] + state["soulHearts"]
            if cur_hp < prev_hp:
                reward += P_LOSE_HEART * (prev_hp - cur_hp)

            # Смерть.
            if state["isDead"] and not self.prev["isDead"]:
                reward += P_DEATH
                done = True

            # Урон врагам: упало их суммарное HP.
            if state["enemyHp"] < self.prev["enemyHp"]:
                reward += R_DAMAGE_ENEMY

            # Зачистка комнаты.
            if state["roomClear"] and not self.prev["roomClear"]:
                reward += R_ROOM_CLEAR

            # Переход в новую комнату.
            if state["roomId"] != self.prev["roomId"]:
                reward += R_NEW_ROOM

            # Новый этаж.
            if state["stage"] > self.prev["stage"]:
                reward += R_NEW_FLOOR

            # Подбор монеты.
            if state["coins"] > self.prev["coins"]:
                reward += R_PICKUP_COIN

            # Слепая стрельба: стреляем, а врагов в комнате НЕТ (точно!).
            if is_shooting and state["enemies"] == 0:
                reward += P_BLIND_SHOT

        self.prev = state
        info["hearts"] = state["hearts"]
        info["enemies"] = state["enemies"]
        return reward, done, info