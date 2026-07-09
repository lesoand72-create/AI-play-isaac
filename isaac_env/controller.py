
# =======================================================================
#  controller.py — эмуляция нажатий клавиш (DirectInput).
# =======================================================================
import pydirectinput

# Убираем встроенную задержку pydirectinput между действиями —
# нам важна скорость, паузами управляем сами.
pydirectinput.PAUSE = 0
# FAILSAFE=False, чтобы курсор в углу экрана НЕ прерывал программу.
pydirectinput.FAILSAFE = False

class Controller:
    """Держит клавиши нажатыми, пока агент этого хочет."""

    def __init__(self):
        # Множество клавиш, которые СЕЙЧАС удерживаются нажатыми.
        self.pressed = set()

    def release_all(self):
        """Отпустить все клавиши (важно при сбросе/остановке)."""
        for key in list(self.pressed):
            pydirectinput.keyUp(key)
        self.pressed.clear()

    def apply(self, keys_to_hold):
        """Сделать так, чтобы были нажаты РОВНО эти клавиши."""
        desired = set(keys_to_hold)

        # 1) Отпускаем те, что больше не нужны.
        for key in list(self.pressed):
            if key not in desired:
                pydirectinput.keyUp(key)
                self.pressed.discard(key)

        # 2) Нажимаем новые, которые ещё не нажаты.
        for key in desired:
            if key not in self.pressed:
                pydirectinput.keyDown(key)
                self.pressed.add(key)