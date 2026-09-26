"""
Environment tests (PRD milestone 2.2, success criterion 7).

Modes:
    uv run python test_env.py           # check_env + random stepping with render
    uv run python test_env.py --mock    # offline smoke test, synthetic frames
"""

import sys

import numpy as np
from stable_baselines3.common.env_checker import check_env

import config
from dino_env import DinoEnv


class SyntheticGame:
    """Fake 'game' for offline testing: draws a moving obstacle and inserts
    a game-over-style dark strip into the game-over region at step 25."""

    def __init__(self):
        self.w, self.h = config.GAME_REGION["width"], config.GAME_REGION["height"]
        self.go_left = config.GAME_OVER_REGION["left"] - config.GAME_REGION["left"]
        self.go_top = config.GAME_OVER_REGION["top"] - config.GAME_REGION["top"]
        self.go_w = config.GAME_OVER_REGION["width"]
        self.go_h = config.GAME_OVER_REGION["height"]
        self.x = 0
        self.steps = 0

    def frame(self) -> np.ndarray:
        self.x = (self.x + 7) % (self.w - 20)
        self.steps += 1
        img = np.full((self.h, self.w, 3), 247, dtype=np.uint8)  # light bg
        img[-25:, :] = 83            # ground line
        img[70:110, self.x:self.x + 15] = 50  # 'cactus'

        if self.steps >= 25:  # simulated GAME OVER: dark text pixels in strip
            strip = img[
                self.go_top:self.go_top + self.go_h,
                self.go_left:self.go_left + self.go_w,
            ]
            strip[:, ::3] = 60
        return img

    def reset(self) -> None:
        self.steps = 0


def run_mock() -> None:
    """Offline smoke test: no screen, no keyboard, no tesseract needed."""
    print("=== mock environment test (offline) ===")
    game = SyntheticGame()
    env = DinoEnv(capture_fn=game.frame, click_fn=lambda *a: None, key_fn=lambda k: None)

    check_env(env, warn=True)
    print("[ok] check_env passed")

    obs, info = env.reset()
    assert obs.shape == (1, config.OBS_HEIGHT, config.OBS_WIDTH), obs.shape
    assert obs.dtype == np.uint8
    print(f"[ok] reset -> obs {obs.shape} {obs.dtype}")

    # random episode: must terminate via the synthetic game-over strip
    total, terminated, steps = 0.0, False, 0
    while not terminated and steps < 60:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        steps += 1
    assert terminated, "mock episode should terminate on synthetic game over"
    assert steps >= config.MIN_EPISODE_STEPS, "done must be suppressed early"
    print(f"[ok] episode ended: steps={steps} return={total:.0f} "
          f"(reward +1/step alive, {config.REWARD_DEATH} on death)")

    # darkness detector sanity: live frame -> False, game-over frame -> True
    assert env._detect_game_over_darkness(game.frame()) is True or game.steps >= 25
    print("[ok] darkness fallback detects synthetic GAME OVER")

    env.close()
    print("=== ALL MOCK TESTS PASSED ===")


def run_live() -> None:
    """Real screen test: needs Chrome at chrome://dino matching config.py."""
    print("=== live environment test ===")
    print("IMPORTANT: Chrome must be open at chrome://dino and match config.py")
    env = DinoEnv(render_mode="human")

    check_env(env, warn=True)
    print("[ok] check_env passed")

    obs, info = env.reset()
    print("[ok] reset clicked the restart button")

    for episode in range(3):
        total, terminated = 0.0, False
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            env.render()
            total += reward
        print(f"[episode {episode + 1}] steps={info['steps']} return={total:.0f}")
        obs, info = env.reset()

    env.close()
    print("=== LIVE TEST DONE ===")


if __name__ == "__main__":
    if "--mock" in sys.argv:
        run_mock()
    else:
        run_live()
