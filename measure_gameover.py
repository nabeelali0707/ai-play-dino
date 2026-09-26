"""
Measure the REAL game-over UI geometry (step 3 refinement).

Instead of trusting css layout constants, this script:
  1. starts the game and waits for the dino to crash (no input),
  2. detects the crash by whole-canvas FREEZE (the game-over screen is
     completely static; gameplay never is),
  3. measures the actual "GAME OVER" text bbox and restart icon position
     from background-deviating pixels (right half of canvas, so the dead
     dino sprite on the left is excluded),
  4. writes the measured GAME_OVER_REGION and RESTART_BUTTON to config.py.

Usage:  .venv/Scripts/python.exe measure_gameover.py
"""

import re
import sys
import time

import cv2
import numpy as np

import config
import win_utils

FREEZE_DIFF = 0.05      # mean |diff| below this = static frame
FREEZE_SECONDS = 1.5    # must stay static this long = game over screen
CRASH_TIMEOUT = 60      # first cactus arrives well within this
GO_TEXT_DELTA = 60      # same text-pixel definition as the env


def main() -> int:
    hwnd, _ = win_utils.pick_dino_window()
    if hwnd is None:
        print("[!] no chrome window; run launch_game.py first")
        return 1
    win_utils.focus_window(hwnd)
    time.sleep(0.3)

    import pydirectinput
    pydirectinput.press("space")  # start the game
    time.sleep(0.5)

    # -- 1. wait for the crash via whole-canvas freeze -----------------------
    print("[*] waiting for the dino to crash (no input)...")
    prev = None
    static_since = None
    t0 = time.time()
    while time.time() - t0 < CRASH_TIMEOUT:
        frame = env_capture()
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
            if prev is not None and prev.shape == gray.shape:
                diff = float(np.mean(np.abs(gray - prev)))
                if diff < FREEZE_DIFF:
                    static_since = static_since or time.time()
                    if time.time() - static_since >= FREEZE_SECONDS:
                        break
                else:
                    static_since = None
            prev = gray
        time.sleep(0.15)
    else:
        print("[!] game never froze within timeout — did it even start?")
        return 1
    print(f"[*] game over detected after {time.time() - t0:.1f}s (frozen screen)")

    # -- 2. measure text and icon from the frozen frame ----------------------
    frame = env_capture()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    H, W = gray.shape
    bg = float(np.median(gray))
    textish = np.abs(gray - bg) >= GO_TEXT_DELTA

    # exclude the dead dino (left side) and any ground texture (below ~88%)
    mask = np.zeros_like(textish, dtype=np.uint8)
    x0 = int(W * 0.25)
    y1 = int(H * 0.88)
    mask[:y1, x0:] = textish[:y1, x0:].astype(np.uint8)

    # 2D connected components; classify by position/size structure:
    #   GAME OVER text: wide blob, horizontally centered, upper-middle
    #   restart icon:   small blob, horizontally centered, below the text
    #   (score text top-right, clouds, etc. fail the center-x test)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    text_cands, icon_cands = [], []
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, k]) for k in range(5))
        if area < 30:
            continue
        nx, ny = centroids[i][0] / W, centroids[i][1] / H
        if 0.30 <= nx <= 0.70 and 0.10 <= ny <= 0.55 and w > 0.15 * W:
            text_cands.append((area, (x, y, w, h)))
        if 0.40 <= nx <= 0.60 and 0.50 <= ny <= 0.85 \
                and w < 0.15 * W and h < 0.2 * H:
            icon_cands.append((area, (x, y, w, h)))
    if not text_cands or not icon_cands:
        print(f"[!] could not identify text/icon components "
              f"(text={text_cands}, icon={icon_cands})")
        cv2.imwrite("debug/gameover_mask.png", mask * 255)
        return 1

    _, (tx0, ty0, tw, th) = max(text_cands)
    _, (ix0, iy0, iw, ih) = max(icon_cands)
    tx1, ty1 = tx0 + tw, ty0 + th
    ix1, iy1 = ix0 + iw, iy0 + ih
    print(f"[*] GAME OVER text bbox: ({tx0},{ty0})..({tx1},{ty1})")
    print(f"[*] restart icon bbox:  ({ix0},{iy0})..({ix1},{iy1})")

    # -- 3. write measured values to config.py -------------------------------
    margin = 6
    go = {
        "left": config.GAME_REGION["left"] + tx0 - margin,
        "top": config.GAME_REGION["top"] + ty0 - margin,
        "width": (tx1 - tx0) + 2 * margin,
        "height": (ty1 - ty0) + 2 * margin,
    }
    restart = (
        config.GAME_REGION["left"] + (ix0 + ix1) // 2,
        config.GAME_REGION["top"] + (iy0 + iy1) // 2,
    )

    with open("config.py", "r", encoding="utf-8") as f:
        cfg = f.read()

    def replace(pattern, repl, text):
        new, n = re.subn(pattern, repl, text, count=1)
        if n != 1:
            raise RuntimeError(f"config.py pattern not found: {pattern}")
        return new

    cfg = replace(
        r"GAME_OVER_REGION\s*=\s*\{[^}]*\}",
        'GAME_OVER_REGION = {"left": %d, "top": %d, "width": %d, "height": %d}'
        % (go["left"], go["top"], go["width"], go["height"]),
        cfg,
    )
    cfg = replace(
        r"RESTART_BUTTON\s*=\s*\([^)]*\)",
        "RESTART_BUTTON = (%d, %d)" % restart,
        cfg,
    )
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(cfg)

    # annotated proof
    ann = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(ann, (tx0, ty0), (tx1, ty1), (0, 0, 255), 2)
    cv2.rectangle(ann, (ix0, iy0), (ix1, iy1), (0, 255, 255), 2)
    cv2.circle(ann, (restart[0] - config.GAME_REGION["left"],
                     restart[1] - config.GAME_REGION["top"]), 8, (255, 0, 0), 2)
    cv2.imwrite("debug/gameover_measured.png", ann)

    import importlib
    importlib.reload(config)
    print("\n=== written to config.py ===")
    print(f"GAME_OVER_REGION = {config.GAME_OVER_REGION}")
    print(f"RESTART_BUTTON   = {config.RESTART_BUTTON}")
    print("[*] proof image: debug/gameover_measured.png")
    print("=== measurement OK ===")

    # leave the game in a clean state for the next step
    pydirectinput.press("space")
    return 0


def env_capture():
    """Capture the configured game region via mss."""
    import mss
    with mss.mss() as sct:
        shot = sct.grab(config.GAME_REGION)
    return np.array(shot)[:, :, :3]


if __name__ == "__main__":
    sys.exit(main())
