"""
Evaluation / inference (PRD 4.3).

Loads a trained checkpoint and runs model.predict() against the live
environment so you can visually confirm the agent plays sensibly.

Usage:
    uv run python play.py                         # latest checkpoint in train/
    uv run python play.py train/dino_model_50000.zip
"""

import glob
import os
import sys
import time

import cv2
from stable_baselines3 import DQN

import config
from dino_env import DinoEnv


def latest_checkpoint() -> str:
    """Newest numbered checkpoint in train/ (falls back to dino_model_final)."""
    candidates = glob.glob(os.path.join(config.TRAIN_DIR, "dino_model_*.zip"))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoints found in train/. Train first: uv run python train.py"
        )

    def step_count(path):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(name.rsplit("_", 1)[1])
        except ValueError:
            return -1  # 'final' sorts lowest, real checkpoints win

    return max(candidates, key=step_count)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else latest_checkpoint()
    print(f"[*] loading {model_path}")
    model = DQN.load(model_path)

    env = DinoEnv(render_mode="human")
    obs, info = env.reset()
    print("[*] playing — press q in the preview window to stop")

    episode, total_reward = 1, 0.0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            env.render()
            total_reward += reward

            if terminated:
                print(
                    f"[episode {episode}] steps={info['steps']} "
                    f"return={total_reward:.0f}"
                )
                episode += 1
                total_reward = 0.0
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\n[*] stopped by user")
    finally:
        time.sleep(0.2)
        env.close()


if __name__ == "__main__":
    main()
