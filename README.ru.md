# ИИ играет в The Binding of Isaac 🎮

[English](README.md) | **Русский**

Нейросеть учится играть в *The Binding of Isaac (Afterbirth+)* с помощью
обучения с подкреплением (Reinforcement Learning, алгоритм **PPO**).

## Как это работает
- **Глаза агента** — захват окна игры (`mss`), кадр обрезается, переводится
  в ч/б и ужимается до 84×84.
- **Руки агента** — эмуляция нажатий клавиш (`pydirectinput`): WASD + стрелки.
- **Учитель (награды)** — Lua-мод пишет точные данные игры (сердца, монеты,
  враги, этаж) в `state.json`, Python читает их и считает награду.
- **Мозг** — PPO с CnnPolicy из `stable-baselines3`.

## Требования
- Windows
- Python 3.12
- The Binding of Isaac: **Afterbirth+** или Repentance (для Lua-мода)
- Видеокарта NVIDIA (опционально, ускоряет обучение)

## Установка
git clone https://github.com/lesoand72-create/AI-play-isaac.git
cd AI-play-isaac
python -m venv .venv
.venvScriptsActivate.ps1
pip install -r requirements.txt


## Запуск
python train.py   # обучение
python play.py    # посмотреть, как играет обученная модель
tensorboard --logdir logs   # графики обучения


## Структура проекта
- `isaac_env/` — окружение Gymnasium (захват, препроцессинг, награды, управление)
- `train.py` / `play.py` — обучение и запуск модели
- `mods/rl_exporter/` — Lua-мод, экспортирующий состояние игры
