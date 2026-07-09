# =======================================================================
#  preprocess.py — препроцессинг кадра: обрезка, серый, ресайз.
# =======================================================================
import cv2            # OpenCV — обработка изображений
import numpy as np

from . import config


def preprocess(frame_bgr):
    """На вход: цветной кадр (H, W, 3). На выход: (84, 84, 1) uint8."""
    h, w = frame_bgr.shape[:2]   # высота и ширина исходного кадра

    # 1) ОБРЕЗКА краёв по долям из config.
    top = int(h * config.CROP_TOP)
    bottom = int(h * (1 - config.CROP_BOTTOM))
    left = int(w * config.CROP_LEFT)
    right = int(w * (1 - config.CROP_RIGHT))
    cropped = frame_bgr[top:bottom, left:right]

    # 2) ОТТЕНКИ СЕРОГО: из 3 каналов (BGR) делаем 1 канал.
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # 3) РЕСАЙЗ до 84x84. INTER_AREA — лучший метод при уменьшении.
    resized = cv2.resize(
        gray,
        (config.FRAME_SIZE, config.FRAME_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # 4) Добавляем ось канала: (84, 84) -> (84, 84, 1),
    #    потому что Gymnasium ждёт изображение с каналом.
    obs = resized[:, :, np.newaxis]

    # Возвращаем как uint8 (0..255). Нормализацию в 0..1 сделает
    # сам stable-baselines3 внутри CnnPolicy.
    return obs.astype(np.uint8)