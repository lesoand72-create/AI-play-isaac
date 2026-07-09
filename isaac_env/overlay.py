# =======================================================================
#  overlay.py — рисует панель "Попытка / Очки" поверх кадра.
#  Используем Pillow, потому что OpenCV не умеет кириллицу.
# =======================================================================
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = None


def _get_font(size=22):
    """Загружаем шрифт с поддержкой кириллицы (один раз)."""
    global _FONT_CACHE
    if _FONT_CACHE is None:
        try:
            # arial.ttf есть почти на всех Windows.
            _FONT_CACHE = ImageFont.truetype("arial.ttf", size)
        except Exception:
            # запасной шрифт (может не поддерживать кириллицу идеально)
            _FONT_CACHE = ImageFont.load_default()
    return _FONT_CACHE


def draw_panel(frame_bgr, attempt, score):
    """Рисует полупрозрачную панель с двумя строками в левом верхнем углу."""
    img = frame_bgr.copy()

    # 1) Полупрозрачная тёмная подложка, чтобы текст читался.
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (300, 95), (0, 0, 0), -1)  # чёрный прямоугольник
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)             # смешиваем 50/50

    # 2) Переходим в Pillow, чтобы нарисовать русский текст.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(24)

    # Первая строка — белая, вторая — зелёная.
    draw.text((22, 16), f"Попытка: {attempt}", font=font, fill=(255, 255, 255))
    draw.text((22, 52), f"Очки: {score:.1f}", font=font, fill=(0, 255, 0))

    # 3) Возвращаемся обратно в формат OpenCV (BGR).
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)