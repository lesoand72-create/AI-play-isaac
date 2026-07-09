# =======================================================================
#  play.py — загружаем обученную модель и смотрим, как она играет.
# =======================================================================
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from isaac_env.env import IsaacEnv


def make_env():
    return IsaacEnv(render_mode="human")


def main():
    print("Переключись на окно игры! Старт через:")
    for i in range(5, 0, -1):
        print(i, "...")
        time.sleep(1)

    # Окружение оборачиваем ТАК ЖЕ, как при обучении (важно: те же обёртки!).
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    # Загружаем сохранённую модель.
    model = PPO.load("models/ppo_isaac", env=env)

    obs = env.reset()
    while True:
        # deterministic=True — модель выбирает лучшее действие без случайности.
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        if dones[0]:            # эпизод закончился — начинаем новый
            obs = env.reset()


if __name__ == "__main__":
    main()