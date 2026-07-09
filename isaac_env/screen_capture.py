# =======================================================================
#  screen_capture.py — делает скриншот окна игры и отдаёт его как массив.
# =======================================================================
import numpy as np      # массивы чисел (изображение = массив)
import mss              # быстрый захват экрана

from . import config    # наши настройки из config.py

# pygetwindow есть только на Windows; аккуратно импортируем.
try:
    import pygetwindow as gw
except Exception:
    gw = None


class ScreenCapture:
    """Отвечает за то, чтобы делать скриншоты именно окна игры."""

    def __init__(self):
        # mss.mss() — объект, через который делаются скриншоты.
        self._sct = mss.mss()
        # Определяем прямоугольник экрана, который будем захватывать.
        self.region = self._find_region()

    def _find_region(self):
        # Если область задана вручную — используем её.
        if config.MANUAL_REGION is not None:
            return config.MANUAL_REGION

        # Иначе ищем окно игры по заголовку.
        if gw is None:
            raise RuntimeError(
                "pygetwindow недоступен. Задай MANUAL_REGION в config.py вручную."
            )
        windows = gw.getWindowsWithTitle(config.GAME_WINDOW_TITLE)
        if not windows:
            raise RuntimeError(
                f"Окно '{config.GAME_WINDOW_TITLE}' не найдено. "
                f"Запусти игру в оконном режиме."
            )
        w = windows[0]  # берём первое подходящее окно
        # Возвращаем словарь с координатами и размером окна.
        return {"top": w.top, "left": w.left, "width": w.width, "height": w.height}

    def grab(self):
        """Делает один скриншот и возвращает его как BGR-массив (H, W, 3)."""
        raw = self._sct.grab(self.region)   # «сырой» скриншот (формат BGRA)
        img = np.array(raw)                 # переводим в массив numpy
        img = img[:, :, :3]                 # отбрасываем альфа-канал -> остаётся BGR
        return img