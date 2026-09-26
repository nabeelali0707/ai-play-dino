"""
Ensure a Chrome window showing chrome://dino exists at a fixed, known
geometry (PRD step 2 - automated window placement, no Selenium needed).

Strategy (learned from live runs on this machine):
  1. If a Chrome window already shows the dino game, use it.
  2. Else launch Chrome with --new-window chrome://dino and grab the new window.
  3. Else navigate a plain Chrome window via Ctrl+L + typed URL.
Then place the window at the fixed geometry (restoring maximized state
first) and focus it. Never touches unrelated Chrome windows beyond
closing stray empty "New tab" app windows.

Usage:  .venv/Scripts/python.exe launch_game.py
"""

import os
import subprocess
import sys
import time

import win_utils

# Fixed placement: top-left corner, canvas-friendly size.
WINDOW = (0, 0, 800, 600)  # left, top, width, height (physical px)

CHROME_CANDIDATES = [
    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
    os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
]

DINO_URL = "chrome://dino/"
MAIN_BROWSER_SUFFIX = "- google chrome"


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "chrome.exe not found; set CHROME_CANDIDATES or install Chrome"
    )


def find_existing_dino_window():
    """(hwnd, title, is_dino) for an already-open window, or (None,)*3.

    Prefers a window whose title mentions the dino game; falls back to the
    main browser window (title carries the '- Google Chrome' suffix), which
    we can navigate to chrome://dino with keystrokes.
    """
    wins = win_utils.find_chrome_windows()
    for hwnd, title in wins:
        if "dino" in title.lower() or "dinosaur" in title.lower():
            return hwnd, title, True
    mains = [(h, t) for h, t in wins if MAIN_BROWSER_SUFFIX in t.lower()]
    if mains:
        return mains[0][0], mains[0][1], False
    if wins:
        return wins[0][0], wins[0][1], False
    return None, None, False


def navigate_to_dino(hwnd) -> bool:
    """Focus window and type the URL into the omnibox."""
    import pydirectinput

    if not win_utils.focus_window(hwnd):
        return False
    time.sleep(0.3)
    pydirectinput.hotkey("ctrl", "l")
    time.sleep(0.4)
    pydirectinput.typewrite("chrome://dino\n", interval=0.02)
    time.sleep(1.5)
    title = win_utils._window_title(hwnd)
    print(f"[*] title after navigation: {title!r}")
    return "dino" in title.lower()


def main():
    left, top, w, h = WINDOW

    # 1) Reuse an existing dino window when possible.
    hwnd, title, is_dino = find_existing_dino_window()
    if hwnd is not None:
        print(f"[*] using existing Chrome window hwnd=0x{hwnd:X} {title!r}")
    else:
        # 2) Launch Chrome with the dino page in a new window.
        chrome = find_chrome()
        print(f"[*] launching {chrome}")
        before = {h_ for h_, _ in win_utils.find_chrome_windows()}
        subprocess.Popen([chrome, "--new-window", DINO_URL],
                         cwd=os.path.dirname(chrome))
        for _ in range(20):  # wait up to 10s for the new window
            time.sleep(0.5)
            new = [(h_, t_) for h_, t_ in win_utils.find_chrome_windows()
                   if h_ not in before]
            if new:
                hwnd, title = new[0]
                print(f"[*] new window hwnd=0x{hwnd:X} {title!r}")
                break
        if hwnd is None:
            print("[!] Chrome window did not appear.")
            sys.exit(1)
        is_dino = "dino" in title.lower()

    # Close stray empty app windows from earlier --app attempts.
    for h_, t_ in win_utils.find_chrome_windows():
        if h_ != hwnd and t_.strip().lower() == "new tab - google chrome":
            print(f"[*] closing stray 'New tab' window 0x{h_:X}")
            win_utils.close_window(h_)

    # 3) Navigate to the game if this window is somewhere else.
    if not is_dino:
        print("[*] navigating to chrome://dino via keyboard")
        is_dino = navigate_to_dino(hwnd)

    # 4) Place at fixed geometry (restore-maximize first) and focus.
    win_utils.place_window(hwnd, left, top, w, h)
    time.sleep(0.5)
    geom = win_utils.get_geometry(hwnd)
    if geom != (left, top, w, h):
        print(f"[!] re-placing (got {geom})")
        win_utils.place_window(hwnd, left, top, w, h)
        geom = win_utils.get_geometry(hwnd)

    focused = win_utils.focus_window(hwnd)

    print("\n=== window geometry (physical px) ===")
    print(f"left={geom[0]} top={geom[1]} width={geom[2]} height={geom[3]}")
    print(f"focused: {focused}")
    print("expected:", WINDOW)
    if geom == WINDOW and focused and is_dino:
        print("=== launch OK: dino window placed, focused, geometry verified ===")
        return 0
    print("=== launch INCOMPLETE "
          f"(geometry_ok={geom == WINDOW}, focused={focused}, on_dino={is_dino}) ===")
    return 2


if __name__ == "__main__":
    sys.exit(main())
