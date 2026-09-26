"""
Interactive calibration (PRD section 5 — coordinates are machine-specific).

Open Chrome at chrome://dino first, position/size the window the way you
will keep it during training, then run:

    uv run python calibrate.py

You'll draw three rectangles with the mouse (drag, then ENTER/SPACE to
confirm, c to skip):

    1. GAME REGION      — playable area: dino, ground, incoming obstacles
    2. GAME-OVER REGION — the strip where "GAME OVER" appears after a crash
    3. RESTART BUTTON   — a tiny box right on the restart icon (center is used)

The script prints ready-to-paste values for config.py and saves an annotated
screenshot to debug/ so you can verify the boxes landed correctly.
"""

import os

import cv2
import mss
import numpy as np
import pytesseract

import config


def grab_screen() -> np.ndarray:
    """Full primary-monitor screenshot as BGR."""
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
    return np.array(shot)[:, :, :3]


def select(label: str, img: np.ndarray):
    """Mouse-select a region; returns (x, y, w, h) or None if skipped."""
    print(f"\n[*] Draw the {label} with the mouse, then press ENTER/SPACE. "
          f"(c = skip)")
    x, y, w, h = cv2.selectROI(f"calibrate: {label}", img, showCrosshair=True)
    cv2.destroyWindow(f"calibrate: {label}")
    if w == 0 or h == 0:
        return None
    print(f"    -> left={x} top={y} width={w} height={h}")
    return int(x), int(y), int(w), int(h)


def test_game_over_ocr(img: np.ndarray, go_region) -> None:
    """Diagnostic: OCR the selected strip and report darkness fraction."""
    x, y, w, h = go_region
    strip = img[y:y + h, x:x + w]

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    dark_frac = float(np.mean(gray < 100))
    print(f"\n[*] game-over strip darkness: {dark_frac:.2f} "
          f"(threshold {config.GAME_OVER_DARKNESS_THRESHOLD}; a live-game "
          f"strip should be well below it)")

    big = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    try:
        text = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip()
        print(f"[*] OCR on strip (live game, so probably empty): {text!r}")
        print("    To fully verify: crash the dino, re-run, and expect 'GAME OVER'-ish text.")
    except pytesseract.TesseractNotFoundError:
        print("[!] Tesseract binary not found — OCR done-detection will rely")
        print("    on the darkness fallback. Install it (see README) or set")
        print("    pytesseract.pytesseract.tesseract_cmd in dino_env.py.")


def main():
    print("=== Dino RL calibration ===")
    print("Chrome must be open at chrome://dino, positioned as it will be during training.")
    input("Press ENTER to take a screenshot...")

    img = grab_screen()

    game = select("GAME REGION", img)
    go = select("GAME-OVER REGION", img)
    btn = select("RESTART BUTTON (tiny box on the icon)", img)

    os.makedirs("debug", exist_ok=True)

    annotated = img.copy()
    if game:
        cv2.rectangle(annotated, game[:2], (game[0] + game[2], game[1] + game[3]), (0, 255, 0), 2)
    if go:
        cv2.rectangle(annotated, go[:2], (go[0] + go[2], go[1] + go[3]), (0, 0, 255), 2)
    if btn:
        center = (btn[0] + btn[2] // 2, btn[1] + btn[3] // 2)
        cv2.circle(annotated, center, 6, (255, 0, 0), 2)
    out = os.path.join("debug", "calibration.png")
    cv2.imwrite(out, annotated)
    print(f"\n[*] annotated screenshot saved to {out}")

    if game and go:
        gx, gy, gw, gh = game
        ox, oy, ow, oh = go
        inside = gx <= ox and gy <= oy and ox + ow <= gx + gw and oy + oh <= gy + gh
        if not inside:
            print("\n[!] WARNING: game-over region is not inside the game region —")
            print("    dino_env.py expects it to overlap the captured frame (PRD 8).")

    print("\n=== paste into config.py ===")
    if game:
        print(f"GAME_REGION      = {{\"left\": {game[0]}, \"top\": {game[1]}, "
              f"\"width\": {game[2]}, \"height\": {game[3]}}}")
    if go:
        print(f"GAME_OVER_REGION = {{\"left\": {go[0]}, \"top\": {go[1]}, "
              f"\"width\": {go[2]}, \"height\": {go[3]}}}")
    if btn:
        print(f"RESTART_BUTTON   = ({btn[0] + btn[2] // 2}, {btn[1] + btn[3] // 2})")

    if go:
        test_game_over_ocr(img, go)

    cv2.destroyAllWindows()
    print("\n[*] After editing config.py, test with: uv run python test_env.py")


if __name__ == "__main__":
    main()
