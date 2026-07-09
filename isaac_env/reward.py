# =======================================================================
#  reward.py — вычисление награды по кадрам экрана.
#  ВНИМАНИЕ: это приблизительные эвристики (см. Часть 3).
# =======================================================================
import cv2
import numpy as np

from . import config

# ----- Числовые значения наград (+) и штрафов (-). Отправная точка. -----
R_SURVIVE_STEP = 0.01    # жив на этом шаге
R_DAMAGE_ENEMY = 0.30    # похоже, нанесли урон врагу
R_ROOM_CLEAR   = 2.00    # комната зачищена
R_NEW_ROOM     = 1.50    # переход в новую комнату
R_PICKUP       = 1.00    # подбор предмета (приближённо)
R_BOSS_DEFEAT  = 20.0    # победа над боссом (приближённо)

P_LOSE_HEART   = -3.0    # потеря сердца / урон
P_DEATH        = -25.0   # смерть
P_IDLE         = -0.05   # бездействие / топтание
P_BLIND_SHOT   = -0.20   # слепая стрельба (врагов рядом нет)


class RewardCalculator:
    """Хранит состояние между кадрами и считает награду."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Сброс перед новым эпизодом."""
        self.prev_gray = None       # предыдущий серый кадр игровой зоны
        self.prev_hearts = None     # сколько было сердец на прошлом шаге
        self.prev_activity = None   # активность врагов на прошлом шаге
        self.static_counter = 0     # сколько шагов подряд ничего не двигалось

    # ----- Вспомогательные методы -----

    def _count_hearts(self, frame_bgr):
        """Грубо оцениваем число сердец по количеству красного в HUD."""
        h, w = frame_bgr.shape[:2]
        r = config.HEARTS_REGION
        y0 = int(h * r["top"])
        y1 = int(h * (r["top"] + r["height"]))
        x0 = int(w * r["left"])
        x1 = int(w * (r["left"] + r["width"]))
        hud = frame_bgr[y0:y1, x0:x1]

        # Переводим в HSV — так проще ловить красный цвет.
        hsv = cv2.cvtColor(hud, cv2.COLOR_BGR2HSV)
        # Красный в HSV живёт в двух диапазонах (в начале и в конце круга оттенков).
        mask1 = cv2.inRange(hsv, (0, 120, 90), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 120, 90), (180, 255, 255))
        mask = mask1 | mask2

        red_ratio = float(np.count_nonzero(mask)) / mask.size
        # Эмпирика: каждые ~0.03 доли красного ≈ одно сердце.
        # ЭТО значение почти наверняка придётся подстроить под свой экран!
        hearts = red_ratio / 0.03
        return hearts

    def _play_area_gray(self, frame_bgr):
        """Вырезаем центральную игровую зону и переводим в серый."""
        h, w = frame_bgr.shape[:2]
        play = frame_bgr[int(h * 0.15):int(h * 0.90),
                         int(w * 0.10):int(w * 0.90)]
        return cv2.cvtColor(play, cv2.COLOR_BGR2GRAY)

    def _activity(self, gray):
        """Сколько пикселей изменилось с прошлого кадра (доля 0..1)."""
        if self.prev_gray is None:
            return 0.0
        diff = cv2.absdiff(gray, self.prev_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return float(np.count_nonzero(thresh)) / thresh.size

    # ----- Главный метод -----

    def compute(self, frame_bgr, is_shooting):
        """Возвращает (reward, done, info) для текущего кадра."""
        reward = 0.0
        done = False
        info = {}

        gray = self._play_area_gray(frame_bgr)

        # (1) Маленькая награда за выживание.
        reward += R_SURVIVE_STEP

        # (2) Сердца: урон и смерть.
        hearts = self._count_hearts(frame_bgr)
        if self.prev_hearts is not None:
            # заметное падение числа сердец = получили урон
            if hearts < self.prev_hearts - 0.4:
                reward += P_LOSE_HEART
            # сердец почти не осталось = смерть, эпизод завершаем
            if hearts <= 0.15:
                reward += P_DEATH
                done = True
        self.prev_hearts = hearts

        # (3) Активность врагов: урон врагу и зачистка комнаты.
        activity = self._activity(gray)
        enemies_present = activity > 0.01
        if self.prev_activity is not None:
            drop = self.prev_activity - activity
            # активность падает, но враги ещё есть -> вероятно, ранили врага
            if enemies_present and drop > 0.005:
                reward += R_DAMAGE_ENEMY
            # была активность и резко пропала -> комната зачищена
            if self.prev_activity > 0.02 and activity < 0.003:
                reward += R_ROOM_CLEAR
        self.prev_activity = activity

        # (4) Переход в новую комнату: сменилась почти вся картинка.
        if self.prev_gray is not None:
            full_diff = cv2.absdiff(gray, self.prev_gray)
            changed = float(np.count_nonzero(full_diff > 40)) / full_diff.size
            if changed > 0.60:
                reward += R_NEW_ROOM
        self.prev_gray = gray

        # (5) Слепая стрельба: стреляет, а врагов рядом нет.
        if is_shooting and not enemies_present:
            reward += P_BLIND_SHOT

        # (6) Бездействие: несколько шагов подряд ничего не двигалось.
        if activity < 0.001:
            self.static_counter += 1
        else:
            self.static_counter = 0
        if self.static_counter > 10:
            reward += P_IDLE

        # Отдаём диагностику наружу (пригодится для панели и логов).
        info["hearts"] = hearts
        info["activity"] = activity
        info["enemies_present"] = enemies_present
        return reward, done, info