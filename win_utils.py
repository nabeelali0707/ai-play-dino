"""
Windows-only window management helpers (ctypes, zero extra deps).

Used to programmatically place and focus the Chrome dino window so the
whole pipeline (launch -> calibrate -> verify -> train) needs no manual
positioning (PRD step 2).
"""

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.windll.user32

# Make coordinate APIs return physical pixels (mss also uses physical px)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


ENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _window_title(hwnd) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _process_name(hwnd) -> str:
    """Executable name owning the window (e.g. 'chrome.exe').

    Needed because Electron apps (VS Code, Slack...) share Chrome's
    'Chrome_WidgetWin_1' window class and would otherwise be picked.
    """
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    k32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wintypes.DWORD(512)
        k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
        return buf.value.rsplit("\\", 1)[-1].lower()
    finally:
        k32.CloseHandle(h)


def find_chrome_windows():
    """Visible top-level windows owned by chrome.exe (class-filtered too)."""
    found = []
    keep = []  # prevent GC of closures during enumeration

    def cb(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls, 256)
            if cls.value == "Chrome_WidgetWin_1" and _process_name(hwnd) == "chrome.exe":
                title = _window_title(hwnd)
                if title:  # skip invisible 0x0 helper windows
                    found.append((hwnd, title))
        return True

    keep.append(ENUMPROC(cb))
    user32.EnumWindows(keep[0], 0)
    return found


def pick_dino_window():
    """Pick the dino app window among chrome.exe windows.

    Preference order:
      1. title mentions dino/dinosaur
      2. title without the '- Google Chrome' suffix (app windows show only
         the page title; the main browser window always has the suffix)
    """
    wins = find_chrome_windows()
    for hwnd, title in wins:
        low = title.lower()
        if "dino" in low or "dinosaur" in low:
            return hwnd, title
    for hwnd, title in wins:
        if "google chrome" not in title.lower():
            return hwnd, title
    if wins:
        return wins[0]
    return None, None


def get_geometry(hwnd):
    """Outer window rect as (left, top, width, height) in physical px."""
    r = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def place_window(hwnd, left, top, width, height) -> None:
    """Move/resize and raise the window.

    Maximized windows ignore SetWindowPos dimensions in practice, so
    restore them first (SW_RESTORE) before setting the geometry.
    """
    SW_RESTORE = 9
    if user32.IsZoomed(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.2)
    user32.SetWindowPos(hwnd, 0, left, top, width, height, 0)
    time.sleep(0.3)


def focus_window(hwnd) -> bool:
    """Bring window to foreground; returns True if it ended up foreground.

    Uses a small ALT-key nudge, the standard workaround for Windows'
    foreground lock when a background process tries to steal focus.
    """
    user32.SetForegroundWindow(hwnd)
    if user32.GetForegroundWindow() == hwnd:
        return True
    # alt-nudge workaround
    user32.keybd_event(0x12, 0, 0, 0)        # ALT down
    user32.SetForegroundWindow(hwnd)
    user32.keybd_event(0x12, 0, 2, 0)        # ALT up
    time.sleep(0.15)
    return user32.GetForegroundWindow() == hwnd


def close_window(hwnd) -> None:
    """Ask a window to close politely (WM_CLOSE), like clicking its X."""
    WM_CLOSE = 0x0010
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
    time.sleep(0.4)


def screen_size():
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
