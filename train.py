"""
Training entrypoint (PRD 4.2).

Builds a TrainAndLoggingCallback that checkpoints the model every
CHECKPOINT_EVERY_STEPS into train/, then runs DQN(CnnPolicy) with
TensorBoard logging into logs/.

Usage (Chrome must be open at chrome://dino and match config.py coords):
    uv run python train.py
Watch progress:
    tensorboard --logdir logs/
"""

import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

import config
from dino_env import DinoEnv


class TrainAndLoggingCallback(BaseCallback):
    """Save a checkpoint every `check_freq` steps (PRD 4.2).

    Real-time training runs are long and fragile (browser crash, window
    focus loss, laptop sleep) — checkpointing caps the loss at N steps.
    """

    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            path = os.path.join(self.save_path, f"dino_model_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                print(f"[checkpoint] saved {path}.zip @ {self.num_timesteps} steps")
        return True


def train(total_timesteps: int = None):
    total_timesteps = total_timesteps or config.TOTAL_TIMESTEPS

    os.makedirs(config.LOG_DIR, exist_ok=True)
    env = DinoEnv()

    model = DQN(
        config.DQN_PARAMS["policy"],
        env,
        tensorboard_log=config.LOG_DIR,
        **{k: v for k, v in config.DQN_PARAMS.items() if k != "policy"},
    )

    callback = TrainAndLoggingCallback(
        check_freq=config.CHECKPOINT_EVERY_STEPS,
        save_path=config.TRAIN_DIR,
    )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    finally:
        # save no matter how training ends (crash, Ctrl+C, completion)
        final_path = os.path.join(config.TRAIN_DIR, "dino_model_final")
        model.save(final_path)
        print(f"[done] final model saved to {final_path}.zip")
        env.close()

    return model


if __name__ == "__main__":
    train()
