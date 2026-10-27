"""Shared configuration for the volatility forecasting project."""

from pathlib import Path

# Reproducible data settings
TICKER = "^GSPC"
START_DATE = "2000-01-01"
END_DATE = "2022-01-01"  # yfinance end date is exclusive
TRAIN_END = "2017-12-31"
VAL_END = "2019-12-31"
TEST_START = "2020-01-01"

# Feature/forecast settings
VOL_WINDOW = 5
LOOKBACK = 30
SEED = 42
GARCH_REFIT_EVERY = 20
GARCH_DIST = "t"

# LSTM settings (used by Anvi's part)
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_BATCH = 32
LSTM_MAX_EPOCHS = 100
LSTM_PATIENCE = 10
N_SEEDS = 3
EPS = 1e-6

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"
MODELS_DIR = RESULTS_DIR / "models"

# Create output directories when the project is first imported.
for directory in (DATA_DIR, RESULTS_DIR, FIG_DIR, DOCS_DIR, TESTS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
