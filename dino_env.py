"""
DinoEnv - a Gymnasium environment for the Chrome offline Dino game.

The agent perceives the game exactly like a human: it only sees pixels from
screen capture (mss) and can only act through key presses (pydirectinput).
No game state is read from code or memory (PRD section 1).

Design notes
------------
* Capture / input backends are injectable so the environment can be smoke
  tested without a live screen (see test_env.py --mock).
* Done detection is dual-method (PRD section 8): pytesseract OCR first, then
  a polarity-agnostic pixel fallback with a static-frame check, because OCR
  on the small game-over strip is unreliable ("GAME" -> "GAHE").
* All pixel thresholds are background-relative (median of the region), so
  both Chrome light and dark themes work.
"""

import time
from typing import Optional

import cv2
import gymnasium as gym
import mss
import numpy as np
import pytesseract
import pydirectinput
from gymnasium import spaces

import config


def _default_tesseract_path() -> Optional[str]:
    """Return the Windows Tesseract path if present, else None."""
    import os
    for p in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(p):
            return p
    return None


_tess = _default_tesseract_path()
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess


class DinoEnv(gym.Env):
    """Pixel-input, keyboard-output Gymnasium env for chrome://dino."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        game_region: Optional[dict] = None,
        game_over_region: Optional[dict] = None,
        restart_button: Optional[tuple] = None,
        render_mode: Optional[str] = None,
        *,
        use_ocr: bool = True,
        capture_fn=None,
        click_fn=None,
        key_fn=None,
    ):
        super().__init__()

        self.game_region = dict(game_region or config.GAME_REGION)
        self.game_over_region = dict(game_over_region or config.GAME_OVER_REGION)
        self.restart_button = tuple(restart_button or config.RESTART_BUTTON)
        self.render_mode = render_mode
        self.use_ocr = use_ocr

        # -- injectable backends (default: real screen + real keyboard) ----
        self._capture_fn = capture_fn or self._capture_screen
        self._click_fn = click_fn or self._click_screen
        self._key_fn = key_fn or self._press_key

        # -- gymnasium spaces (PRD 4.1) ------------------------------------
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, config.OBS_HEIGHT, config.OBS_WIDTH),
            dtype=np.uint8,
        )
        # actions: 0 = jump (space), 1 = duck (down), 2 = no-op
        self.action_space = spaces.Discrete(3)

        # -- runtime state --------------------------------------------------
        self._sct: Optional[mss.base.MSSBase] = None
        self._frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self._steps_this_episode = 0
        self._prev_go_gray: Optional[np.ndarray] = None  # static-frame check

    # ----------------------------------------------------------- screen I/O
    def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture the game region -> BGR ndarray, or None on failure."""
        try:
            if self._sct is None:
                self._sct = mss.mss()
            shot = self._sct.grab(self.game_region)
        except Exception:
            return None
        return np.array(shot)[:, :, :3]  # mss gives BGRA; drop alpha

    @staticmethod
    def _click_screen(x: int, y: int) -> None:
        pydirectinput.click(x, y)

    @staticmethod
    def _press_key(key: Optional[str]) -> None:
        if key is None:
            return
        pydirectinput.press(key)

    # ------------------------------------------------------------ pipeline
    def get_observation(self) -> np.ndarray:
        """mss capture -> grayscale -> resize -> add channel dim (PRD 4.1)."""
        frame = self._capture_fn()
        if frame is None:
            return self._frame  # keep last known frame on capture failure

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray, (config.OBS_WIDTH, config.OBS_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        obs = np.expand_dims(resized, axis=0)  # (1, H, W)
        self._frame = obs
        return obs

    # ------------------------------------------------------- done detection
    def _go_strip(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Crop the game-over strip out of a game-region frame, or None."""
        left = self.game_over_region["left"] - self.game_region["left"]
        top = self.game_over_region["top"] - self.game_region["top"]
        w, h = self.game_over_region["width"], self.game_over_region["height"]
        H, W = frame.shape[:2]
        if not (0 <= left and 0 <= top and left + w <= W and top + h <= H):
            return None  # mis-calibrated region
        return frame[top:top + h, left:left + w]

    @staticmethod
    def _gray_i16(strip: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.int16)

    @staticmethod
    def _text_fraction(gray_i16: np.ndarray) -> float:
        """Fraction of text-like pixels (far from the strip's own bg median)."""
        bg = float(np.median(gray_i16))
        return float(np.mean(np.abs(gray_i16 - bg) >= config.GO_TEXT_DELTA))

    def get_done(self) -> bool:
        """Dual done detection (PRD 4.1 / 8): OCR first, pixel fallback.

        Suppressed for the first MIN_EPISODE_STEPS of an episode so the
        previous episode's game-over text cannot cause a false positive
        right after reset().
        """
        if self._steps_this_episode < config.MIN_EPISODE_STEPS:
            self._prev_go_gray = None
            return False

        frame = self._capture_fn()
        if frame is None:
            return False
        strip = self._go_strip(frame)
        if strip is None:
            return False

        gray = self._gray_i16(strip)

        # Fast path: plain background -> nothing to detect. This also makes
        # the static-frame check self-consistent (we only compare frames
        # that contain something text-like).
        if self._text_fraction(gray) < config.GO_TEXT_FRACTION_THRESHOLD:
            self._prev_go_gray = None
            return False

        if self.use_ocr and self._detect_game_over_ocr(strip):
            return True
        return self._detect_game_over_static(gray)

    def _detect_game_over_ocr(self, strip: np.ndarray) -> bool:
        """OCR the strip; fuzzy G-A-M prefix match (OCR reads GAME as GAHE,
        CAME, GAN... but the leading G-A-M is nearly always intact)."""
        big = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        # Polarity-safe binarization (dark mode renders LIGHT text on dark):
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(np.mean(thresh)) > 127:  # mostly white -> make text dark
            thresh = cv2.bitwise_not(thresh)

        try:
            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            ).upper()
        except Exception:
            return False  # tesseract missing / crashed -> pixel fallback

        return any(token in text for token in ("GAM", "GAH", "CAH", "GAN"))

    def _detect_game_over_static(self, gray_i16: np.ndarray) -> bool:
        """Polarity-agnostic pixel fallback.

        "GAME OVER" text is large AND static; a passing pterodactyl is
        text-like but moving. Require the strip to be unchanged vs the
        previous text-like capture.
        """
        static = True
        if self._prev_go_gray is not None and \
                self._prev_go_gray.shape == gray_i16.shape:
            mean_diff = float(np.mean(np.abs(gray_i16 - self._prev_go_gray)))
            static = mean_diff <= config.GO_STATIC_MAX_DIFF
        self._prev_go_gray = gray_i16
        return static

    # ----------------------------------------------------------- gym API
    def step(self, action):
        """Send key -> capture -> check done -> reward (PRD 4.1)."""
        key = config.KEY_MAP.get(int(action))
        self._key_fn(key)

        time.sleep(config.STEP_SLEEP)
        obs = self.get_observation()
        self._steps_this_episode += 1

        terminated = bool(self.get_done())
        truncated = False
        reward = config.REWARD_ALIVE if not terminated else config.REWARD_DEATH

        info = {"steps": self._steps_this_episode}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Click the restart button and return a fresh observation (PRD 4.1)."""
        super().reset(seed=seed)

        self._click_fn(*self.restart_button)
        time.sleep(0.5)  # let the game restart and the dino respawn
        obs = self.get_observation()
        self._steps_this_episode = 0
        self._prev_go_gray = None

        info = {"steps": 0}
        return obs, info

    def render(self):
        """Debug preview (PRD 4.1, optional). Close with 'q' or the window X."""
        if self.render_mode != "human":
            return
        cv2.imshow("DinoEnv observation", self._frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()
            raise KeyboardInterrupt("render window closed")

    def close(self):
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
        cv2.destroyAllWindows()
