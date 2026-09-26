# AI Plays Chrome Dino 🦖

A DQN reinforcement-learning agent that learns to play the Chrome offline
Dino game by **watching the screen** (mss screen capture) and **pressing
keys** (pydirectinput) — no game-state injection, pure pixel input and
keyboard output.

## How it works

```
screen ──mss──> grayscale 100x83 frame ──> DQN (CnnPolicy) ──> space / down / no-op
                (game-over text OCR'd via pytesseract to end episodes)
```

- **Env:** `dino_env.py` — gymnasium `Env` with `Discrete(3)` actions
  {0: jump, 1: duck, 2: no-op} and `Box(0,255,(1,83,100))` observations.
- **Training:** `train.py` — SB3 DQN with checkpoint callback every 10k steps.
- **Inference:** `play.py` — loads a checkpoint and plays live.

## Setup (Windows)

```powershell
# 1. Python 3.11 venv (SB3-compatible; system 3.14 is too new for the stack)
uv venv .venv --python 3.11
uv pip install -r requirements.txt

# 2. Install Tesseract OCR binary (PRD: system binary required, not just pip pkg)
winget install --id UB-Mannheim.TesseractOCR
#    default install path: C:\Program Files\Tesseract-OCR\tesseract.exe
```

> No GPU required (PRD section 2). Observations are tiny; the bottleneck is
> real-time game interaction.

## Calibration (required before first run!)

All coordinates live in `config.py` and are **specific to your screen and
browser window position**. Any time you move/resize the window or change
resolution, recalibrate:

```powershell
uv run python calibrate.py
```

Then paste the printed values into `config.py`:

- `GAME_REGION` – the playable area (dino + track + incoming obstacles)
- `GAME_OVER_REGION` – strip containing the "GAME OVER" text
- `RESTART_BUTTON` – the restart icon to click on episode end

## Test the environment

```powershell
uv run python test_env.py        # check_env + random actions with preview
```

## Train

```powershell
# 1. Open Chrome at chrome://dino, position the window to match your calibration
# 2. Make sure the tab is focused and unobstructed (PRD section 8)
uv run python train.py
# watch progress:  tensorboard --logdir logs/
```

Checkpoints are written to `train/dino_model_<steps>.zip` every 10k steps, so
a crash never loses more than 10k steps of real-time play.

## Play / evaluate

```powershell
uv run python play.py            # plays with latest checkpoint
uv run python play.py train/dino_model_100000.zip
```

Press `q` in the terminal to stop.

## Success criteria (PRD section 7)

- Full step → observe → act → reset loop with no manual intervention
- Long training runs survive crashes via checkpointing
- Trained agent beats a random-action baseline on survival time
- Done-detection reliably triggers `reset()` shortly after game over
