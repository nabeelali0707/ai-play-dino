"""
Auto-calibration (step 3 of the automated pipeline).

Detects the game canvas by its pixel signature instead of asking the user
to crop a screenshot. Works in BOTH Chrome light and dark themes:

  1. Capture the window content area.
  2. background = per-frame median gray (works for #f7f7f7 AND #202124).
  3. Ground line = the row where the largest fraction of pixels deviate
     strongly from the background (98%+ coverage, full canvas width).
  4. Canvas geometry from known css layout: canvas is 600x150 css px wide,
     ground line sits ~127 css px from the canvas top; scale = width/600.
  5. Derive GAME_REGION (canvas), GAME_OVER_REGION (central text strip),
     RESTART_BUTTON (centered icon below the text); validate on the live
     frame; write results directly into config.py.

Usage:  .venv/Scripts/python.exe auto_calibrate.py
"""

import os
import re
import time

import cv2
import mss
import numpy as np

import win_utils

WINDOW = (0, 0, 800, 600)  # must match launch_game.py

# Content-area inset for a normal Windows window border/title bar.
TITLE_BAR_PX = 31
BORDER_PX = 8

# Dino canvas css layout constants (stable across themes; scale with zoom).
CANVAS_CSS_W, CANVAS_CSS_H = 600.0, 150.0
GROUND_CSS_Y = 127.0          # ground line top, css px from canvas top
GO_TEXT_CSS = (40.0, 80.0)    # 'GAME OVER' text vertical span in css px
RESTART_CSS_Y = 87.5          # restart icon center, css px from canvas top

GROUND_DEV = 60        # px deviating this much from bg count as ground/text


def grab(left, top, width, height) -> np.ndarray:
    with mss.mss() as sct:
        shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return np.array(shot)[:, :, :3]


def detect_ground(gray: np.ndarray):
    """Return (ground_y, coverage) or None. Polarity-agnostic."""
    bg = float(np.median(gray))
    dev_frac = np.mean(np.abs(gray.astype(np.int16) - bg) > GROUND_DEV, axis=1)
    y = int(np.argmax(dev_frac))
    return (y, float(dev_frac[y])) if dev_frac[y] > 0.8 else None


def text_fraction(strip_gray: np.ndarray) -> float:
    """Fraction of text-like pixels (far from strip's own background)."""
    bg = float(np.median(strip_gray))
    return float(np.mean(np.abs(strip_gray.astype(np.int16) - bg) > GROUND_DEV))


def main() -> int:
    hwnd, title = win_utils.pick_dino_window()
    if hwnd is None:
        print("[!] no chrome window; run launch_game.py first")
        return 1
    wx, wy, ww, wh = win_utils.get_geometry(hwnd)
    if (wx, wy, ww, wh) != WINDOW:
        print(f"[*] window at {wx, wy, ww, wh}; re-placing to {WINDOW}")
        win_utils.place_window(hwnd, *WINDOW)
        win_utils.focus_window(hwnd)
        time.sleep(0.5)
        wx, wy, ww, wh = win_utils.get_geometry(hwnd)
        if (wx, wy, ww, wh) != WINDOW:
            print(f"[!] could not fix placement (still {wx, wy, ww, wh})")
            return 1

    cx, cy = wx + BORDER_PX, wy + TITLE_BAR_PX
    cw, ch = ww - 2 * BORDER_PX, wh - TITLE_BAR_PX - BORDER_PX
    print(f"[*] window {wx, wy, ww, wh} -> content region ({cx}, {cy}, {cw}, {ch})")

    img = grab(cx, cy, cw, ch)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/calib_content.png", img)

    det = detect_ground(gray)
    if det is None:
        print("[!] ground line not found. Is the game page visible? "
              "Screenshot: debug/calib_content.png")
        return 1
    ground_y, cov = det
    scale = cw / CANVAS_CSS_W
    print(f"[*] ground line y={ground_y} (coverage {cov:.0%}); "
          f"scale {scale:.3f}x vs css")

    # Canvas vertical span in content coords.
    canvas_top = int(round(ground_y - GROUND_CSS_Y * scale))
    canvas_h = int(round(CANVAS_CSS_H * scale))
    canvas_bot = canvas_top + canvas_h
    if canvas_top < 0 or canvas_bot > ch:
        print(f"[!] derived canvas span {canvas_top}..{canvas_bot} outside "
              f"content (0..{ch}) — window too small or layout changed")
        return 1

    # Screen-space regions --------------------------------------------------
    game = {"left": cx, "top": cy + canvas_top, "width": cw, "height": canvas_h}
    go = {
        "left": cx + int(cw * 0.30),
        "top": cy + canvas_top + int(GO_TEXT_CSS[0] * scale),
        "width": int(cw * 0.40),
        "height": int((GO_TEXT_CSS[1] - GO_TEXT_CSS[0]) * scale),
    }
    restart = (cx + cw // 2,
               cy + canvas_top + int(RESTART_CSS_Y * scale))

    # Validate on the live game: press Space to guarantee the game is
    # running (starts a new game / restarts after a crash / harmless jump),
    # then sample the strip over time. While alive, text-fraction stays low
    # except for the instant a pterodactyl crosses the strip (moving, rare).
    import pydirectinput

    win_utils.focus_window(hwnd)
    time.sleep(0.3)
    pydirectinput.press("space")
    time.sleep(1.0)

    gx, gy = go["left"] - cx, go["top"] - cy
    samples = []
    for i in range(8):
        s = gray[gy:gy + go["height"], gx:gx + go["width"]]
        samples.append(text_fraction(s))
        time.sleep(0.3)
    alive_samples = sum(1 for tf in samples if tf < 0.08)
    print(f"[*] strip text-fraction over 8 samples: "
          f"{['%.3f' % t for t in samples]} -> alive in {alive_samples}/8")
    if alive_samples < 6:
        print("[!] strip shows persistent text — GAME OVER likely on screen "
              "or regions are wrong; refusing to write bad calibration")
        return 1

    # Write config.py -------------------------------------------------------
    with open("config.py", "r", encoding="utf-8") as f:
        cfg = f.read()

    def replace(pattern, repl, text):
        new, n = re.subn(pattern, repl, text, count=1)
        if n != 1:
            raise RuntimeError(f"config.py pattern not found: {pattern}")
        return new

    cfg = replace(
        r"GAME_REGION\s*=\s*\{[^}]*\}",
        'GAME_REGION = {"left": %d, "top": %d, "width": %d, "height": %d}'
        % (game["left"], game["top"], game["width"], game["height"]),
        cfg,
    )
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

    # Annotated screenshot for visual confirmation --------------------------
    ann = img.copy()
    cv2.rectangle(ann, (0, canvas_top), (cw - 1, canvas_bot), (0, 255, 0), 2)
    cv2.rectangle(
        ann,
        (go["left"] - cx, go["top"] - cy),
        (go["left"] - cx + go["width"], go["top"] - cy + go["height"]),
        (0, 0, 255),
        2,
    )
    cv2.circle(ann, (restart[0] - cx, restart[1] - cy), 8, (255, 0, 0), 2)
    cv2.imwrite("debug/calibration_auto.png", ann)

    print("\n=== written to config.py ===")
    print(f"GAME_REGION      = {game}")
    print(f"GAME_OVER_REGION = {go}")
    print(f"RESTART_BUTTON   = {restart}")
    print("[*] annotated proof: debug/calibration_auto.png")
    print("=== auto-calibration OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
