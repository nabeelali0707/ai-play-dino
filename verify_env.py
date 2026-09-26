"""
Environment verification (step 4 of the automated pipeline).

Runs against the REAL screen and validates programmatically, exiting
non-zero on any failure so the pipeline can stop before training:

  1. capture sanity     — frames not blank/black, enough detail
  2. observation flow   — consecutive frames differ (game is animating)
  3. forced game-over   — do nothing until the dino crashes; done must
                          trigger within a deadline
  4. reset              — restart click clears the game-over screen and
                          the game runs again
  5. reward signal      — +1 per alive step observed

Usage:  .venv/Scripts/python.exe verify_env.py
"""

import sys
import time

import cv2
import numpy as np

import config
from dino_env import DinoEnv

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    env = DinoEnv()

    # -- 1. capture sanity ---------------------------------------------------
    obs = env.get_observation()
    gray = obs[0]
    check("capture: not blank", float(np.mean(gray)) > 5 and float(np.mean(gray)) < 250,
          f"mean={np.mean(gray):.1f}")
    check("capture: has detail", int(np.unique(gray).size) > 8,
          f"unique values={np.unique(gray).size}")

    # -- 2. observation flow (game animating) --------------------------------
    import pydirectinput
    import win_utils
    hwnd, title = win_utils.pick_dino_window()
    if hwnd is None:
        print("[FAIL] no chrome window found")
        return 1
    win_utils.focus_window(hwnd)
    time.sleep(0.3)
    pydirectinput.press("space")  # ensure the game is running
    time.sleep(1.0)

    obs1 = env.get_observation().copy()
    time.sleep(0.5)
    obs2 = env.get_observation()
    diff = float(np.mean(np.abs(obs1.astype(int) - obs2.astype(int))))
    # The dino + scrolling ground occupy a small part of the frame, so even
    # a healthy game yields a small mean difference; 0.2 separates live from
    # frozen (verified live: ~0.4 while running, ~0.0 while stopped).
    check("observation: frames change while running", diff > 0.2, f"mean diff={diff:.2f}")

    # -- 3. forced game over: stop acting, dino crashes on first cactus ------
    print("[*] forcing game over (no input; dino will hit the first cactus)")
    deadline = time.time() + 20
    done = False
    env._steps_this_episode = config.MIN_EPISODE_STEPS  # enable done detection
    env._prev_go_gray = None
    while time.time() < deadline:
        if env.get_done():
            done = True
            break
        time.sleep(0.2)
    check("done detection: triggers after crash", done,
          f"within {20 - (deadline - time.time()):.1f}s")

    # -- 4. reset restarts the game -------------------------------------------
    obs, info = env.reset()
    time.sleep(1.0)  # game-over text should be gone now
    frame = env._capture_fn()
    go = config.GAME_OVER_REGION
    strip = frame[
        go["top"] - config.GAME_REGION["top"]:
        go["top"] - config.GAME_REGION["top"] + go["height"],
        go["left"] - config.GAME_REGION["left"]:
        go["left"] - config.GAME_REGION["left"] + go["width"],
    ]
    gray_s = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.int16)
    bg = float(np.median(gray_s))
    tf = float(np.mean(np.abs(gray_s - bg) >= config.GO_TEXT_DELTA))
    check("reset: game-over text cleared", tf < config.GO_TEXT_FRACTION_THRESHOLD,
          f"text fraction={tf:.3f}")

    time.sleep(0.5)
    a = env.get_observation().copy()
    time.sleep(0.5)
    b = env.get_observation()
    moving = float(np.mean(np.abs(a.astype(int) - b.astype(int))))
    check("reset: game running again", moving > 0.2, f"mean diff={moving:.2f}")

    # -- 5. reward signal ------------------------------------------------------
    env._steps_this_episode = 0
    _, reward, terminated, _, _ = env.step(config.ACTION_NOOP)
    check("reward: +1 while alive", reward == config.REWARD_ALIVE, f"reward={reward}")

    env.close()

    print()
    if FAILURES:
        print(f"=== VERIFICATION FAILED: {FAILURES} ===")
        return 1
    print("=== ALL VERIFICATION CHECKS PASSED — environment ready for training ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
