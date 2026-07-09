# =======================================================================
#  train.py — запуск обучения агента PPO.
# =======================================================================
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from isaac_env.env import IsaacEnv

MODELS_DIR = "models"   # куда сохранять модель
LOGS_DIR = "logs"       # куда писать логи для TensorBoard
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def make_env():
    """Фабрика окружения (нужна для VecEnv)."""
    return IsaacEnv(render_mode="human")


def main():
    # Обратный отсчёт: успей переключиться на окно игры!
    print("Переключись на окно игры! Старт через:")
    for i in range(5, 0, -1):
        print(i, "...")
        time.sleep(1)

    # 1) Оборачиваем наше окружение в векторное (так требует SB3).
    env = DummyVecEnv([make_env])
    # 2) VecMonitor — считает статистику эпизодов (награда, длина).
    env = VecMonitor(env)
    # 3) VecFrameStack — склеивает 4 последних кадра (динамика движения).
    env = VecFrameStack(env, n_stack=4)

    # 4) Создаём модель PPO со свёрточной политикой CnnPolicy.
    model = PPO(
        policy="CnnPolicy",     # сеть, которая "умеет смотреть" на картинки
        env=env,
        verbose=1,              # печатать прогресс в консоль
        tensorboard_log=LOGS_DIR,
        n_steps=512,            # сколько шагов собирать перед обновлением
        batch_size=128,         # размер мини-батча
        learning_rate=2.5e-4,   # скорость обучения
        gamma=0.99,             # насколько ценим будущие награды
        device="auto",          # авто: GPU если есть, иначе CPU
    )

    TOTAL_TIMESTEPS = 100_000   # общее число шагов обучения (для старта)

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    finally:
        # Сохраняем модель даже если прервали обучение (Ctrl+C).
        save_path = os.path.join(MODELS_DIR, "ppo_isaac")
        model.save(save_path)
        print(f"Модель сохранена: {save_path}.zip")
        env.close()


if __name__ == "__main__":
    main()