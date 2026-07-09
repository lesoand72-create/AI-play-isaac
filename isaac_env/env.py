# =======================================================================
#  env.py — окружение Gymnasium для The Binding of Isaac.
# =======================================================================
import time

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import config
from .screen_capture import ScreenCapture
from .preprocess import preprocess
from .controller import Controller
from .reward_modded import ModdedRewardCalculator
from .overlay import draw_panel

# --- Список всех возможных действий ---
# Каждое действие = пара (движение, стрельба).
# None означает "ничего не нажимать" по этой оси.
MOVES = [None, config.KEY_UP, config.KEY_DOWN, config.KEY_LEFT, config.KEY_RIGHT]
SHOOTS = [None, config.SHOOT_UP, config.SHOOT_DOWN, config.SHOOT_LEFT, config.SHOOT_RIGHT]

ACTIONS = []
for _move in MOVES:
    for _shoot in SHOOTS:
        ACTIONS.append((_move, _shoot))
# Итого 5 * 5 = 25 действий (индексы 0..24).


class IsaacEnv(gym.Env):
    """Окружение: делает шаг игры, считает награду, рисует панель."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Компоненты: глаза, руки, учитель.
        self.capture = ScreenCapture()
        self.controller = Controller()
        self.reward_calc = ModdedRewardCalculator()
        # Пространство действий: 25 дискретных вариантов.
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Пространство наблюдений: картинка 84x84x1, значения 0..255.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(config.FRAME_SIZE, config.FRAME_SIZE, 1),
            dtype=np.uint8,
        )

        # Счётчики для панели.
        self.attempt = 0            # номер попытки (эпизода)
        self.episode_score = 0.0    # накопленные очки за текущий эпизод
        self.steps = 0              # сколько шагов сделано в этом эпизоде
        self._last_frame = None     # последний цветной кадр (для панели)

    def _get_obs(self):
        """Сделать скриншот и превратить в наблюдение 84x84x1."""
        frame = self.capture.grab()
        self._last_frame = frame
        return preprocess(frame)

    def reset(self, *, seed=None, options=None):
        """Начало нового эпизода (попытки)."""
        super().reset(seed=seed)
        self.controller.release_all()   # отпускаем все клавиши
        self.reward_calc.reset()        # сбрасываем "учителя"
        self.attempt += 1               # +1 к номеру попытки
        self.episode_score = 0.0
        self.steps = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """Один шаг: применяем действие, ждём, считаем награду."""
        # 1) Расшифровываем действие в клавиши.
        move, shoot = ACTIONS[int(action)]
        keys = []
        if move is not None:
            keys.append(move)
        if shoot is not None:
            keys.append(shoot)
        self.controller.apply(keys)

        # 2) Даём игре немного времени отреагировать.
        time.sleep(config.STEP_DELAY)

        # 3) Смотрим на новый кадр.
        frame = self.capture.grab()
        self._last_frame = frame
        obs = preprocess(frame)

        # 4) Считаем награду.
        is_shooting = shoot is not None
        reward, done, rinfo = self.reward_calc.compute(is_shooting)

        # 5) Обновляем счётчики.
        self.episode_score += reward
        self.steps += 1

        terminated = done                          # эпизод закончился "по правилам" (смерть)
        truncated = self.steps >= config.MAX_STEPS # или по лимиту шагов

        # 6) Рисуем панель, если нужно.
        if self.render_mode == "human":
            self.render()

        # 7) Возвращаем всё по стандарту Gymnasium (5 значений).
        info = dict(rinfo)
        info["attempt"] = self.attempt
        info["episode_score"] = self.episode_score
        return obs, reward, terminated, truncated, info

    def render(self):
        """Показывает окно-монитор с кадром и панелью "Попытка / Очки"."""
        if self._last_frame is None:
            return
        # Уменьшаем кадр до удобного размера окна.
        disp = cv2.resize(self._last_frame, (640, 360))
        # Рисуем панель поверх.
        disp = draw_panel(disp, self.attempt, self.episode_score)
        cv2.imshow("Isaac RL — монитор", disp)
        # Делаем окно поверх остальных (работает не на всех системах).
        try:
            cv2.setWindowProperty("Isaac RL — монитор", cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        cv2.waitKey(1)   # обязательно, иначе окно "зависнет"

    def close(self):
        """Аккуратно завершаем: отпускаем клавиши и закрываем окна."""
        self.controller.release_all()
        cv2.destroyAllWindows()