"""
Central configuration for the Dino RL project.

ALL screen coordinates are calibration-dependent (PRD section 5). Use
calibrate.py to find the correct values for your monitor / browser window,
then paste them below. If training suddenly stops working, coordinates
drift is the #1 suspect.
"""

# ---------------------------------------------------------------------------
# Screen regions (absolute pixels, primary monitor, mss format)
# mss region dict: {"left": int, "top": int, "width": int, "height": int}
# ---------------------------------------------------------------------------

# Main game viewport inside the Chrome window (the playable area with the
# dino, cacti and pterodactyls). Captured frames are cropped from here.
GAME_REGION = {"left": 100, "top": 250, "width": 660, "height": 150}

# Small strip containing the "GAME OVER" text (used for done detection).
GAME_OVER_REGION = {"left": 350, "top": 280, "width": 200, "height": 50}

# Fixed screen coordinate of the restart button in the game-over screen.
# reset() clicks this point to start a new episode.
RESTART_BUTTON = (400, 390)

# ---------------------------------------------------------------------------
# Actions (index -> meaning), Discrete(3)
# ---------------------------------------------------------------------------
ACTION_JUMP = 0
ACTION_DUCK = 1
ACTION_NOOP = 2
KEY_MAP = {
    ACTION_JUMP: "space",   # jump / start game
    ACTION_DUCK: "down",    # duck under pterodactyls
    ACTION_NOOP: None,      # do nothing
}

# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------
OBS_WIDTH = 100   # PRD: ~83x100 grayscale; stored as (1, 83, 100)
OBS_HEIGHT = 83

# ---------------------------------------------------------------------------
# Done detection
# ---------------------------------------------------------------------------
# OCR is fragile (PRD 5: "most fragile component"). We OCR first, then fall
# back to a pixel-darkness check. GAME_OVER_REGION is mostly light gray
# background; when "GAME OVER" appears the mean darkness rises sharply.
GAME_OVER_DARKNESS_THRESHOLD = 0.30  # fraction of "dark" pixels in region

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
# Wall-clock sleep after each step. The game is real-time; this pace keeps
# frame-to-frame action changes meaningful. (PRD 5: real-time cost is a
# fixed cost by design, not something to optimize away.)
STEP_SLEEP = 0.05  # seconds -> ~20 steps/second

# ---------------------------------------------------------------------------
# Training (DQN per PRD 4.2)
# ---------------------------------------------------------------------------
LOG_DIR = "logs/"
TRAIN_DIR = "train/"
CHECKPOINT_EVERY_STEPS = 10_000
TOTAL_TIMESTEPS = 500_000

DQN_PARAMS = dict(
    policy="CnnPolicy",
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=64,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.3,
    exploration_final_eps=0.02,
    gamma=0.99,
    verbose=1,
)
