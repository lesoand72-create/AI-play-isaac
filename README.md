# AI plays The Binding of Isaac 🎮

**English** | [Русский](README.ru.md)

A neural network learns to play *The Binding of Isaac (Afterbirth+)* using
Reinforcement Learning (the **PPO** algorithm).

## How it works
- **Agent's eyes** — captures the game window (`mss`); the frame is cropped,
  converted to grayscale, and resized to 84×84.
- **Agent's hands** — key emulation (`pydirectinput`): WASD + arrow keys.
- **The teacher (rewards)** — a Lua mod writes accurate game data (hearts,
  coins, enemies, floor) to `state.json`; Python reads it and computes the reward.
- **The brain** — PPO with a CnnPolicy from `stable-baselines3`.

## Requirements
- Windows
- Python 3.12
- The Binding of Isaac: **Afterbirth+** or Repentance (for the Lua mod)
- NVIDIA GPU (optional, speeds up training)

## Installation
git clone https://github.com/lesoand72-create/AI-play-isaac.git
cd AI-play-isaac
python -m venv .venv
.venvScriptsActivate.ps1
pip install -r requirements.txt

## Usage
python train.py   # train the agent
python play.py    # watch the trained model play
tensorboard --logdir logs   # training charts

## Project structure
- `isaac_env/` — Gymnasium environment (capture, preprocessing, rewards, control)
- `train.py` / `play.py` — training and running the model
- `mods/rl_exporter/` — Lua mod that exports the game state
