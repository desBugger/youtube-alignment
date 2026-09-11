"""Single source of truth for every parameter, window, and seed.

Nothing in this pipeline reads a magic number from anywhere else.
"""
from datetime import datetime
from pathlib import Path

import pytz

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

# --- Seeds -----------------------------------------------------------------
# langdetect is non-deterministic unless its factory seed is fixed. This was
# unset in the original analysis, which is why the language filter could not be
# reproduced exactly. It is fixed here.
SEED = 0

# --- Accounts --------------------------------------------------------------
# Account ids are researcher-assigned pseudonyms for accounts we created.
# They are retained (not stripped) so that per-account variation and the
# collection interruptions documented in the paper stay independently checkable.
CONDITIONS = {
    "watch":    ["watchgamer2012", "watch2gamer2012", "watch3gamer2012"],
    "like":     ["likegamer2012", "like2gamer2012", "like3gamer2012"],
    "notInt":   ["dislikegamer2012", "dislike2gamer2012", "dislike3gamer2012"],
    "combined": ["combinedgamer2012", "combined2gamer2012", "combined3gamer2012"],
}
CONDITION_ORDER = ["watch", "like", "notInt", "combined"]

# dislike3gamer2012 stopped returning video recommendations after 28 May 2025.
# It contributes 40 of 1,346 not-interested records (3.0%). The paper reports
# that condition as n=2; set to False to include it as a sensitivity check.
EXCLUDE_FAILED_ACCOUNT = False
FAILED_ACCOUNT = "dislike3gamer2012"

BST = pytz.timezone("Europe/London")


def _utc(y, m, d, hh, mm):
    return BST.localize(datetime(y, m, d, hh, mm)).astimezone(pytz.utc)


# --- Analysis windows ------------------------------------------------------
# Persona creation ran across days 1-2 (26-27 May). Recommendations collected
# before the end of the day-2 session are excluded as warm-up.
#
# NOTE: the original notebook referenced an undefined `cutoffUTC` here and could
# not run as saved. The intended value is this one.
WARMUP_END = _utc(2025, 5, 27, 12, 25)

# The original analysis applied a *different* end cutoff per condition, and none
# at all to notInt, which gave that condition ~18h more observation than the
# others. PUBLISHED reproduces those cutoffs exactly (for verifying the paper's
# reported numbers); HARMONISED applies one common end to all four conditions.
WINDOW_MODE = "published"  # "published" | "harmonised"

PUBLISHED_END = {
    "watch":    _utc(2025, 6, 5, 22, 0),
    "like":     _utc(2025, 6, 5, 10, 0),
    "notInt":   None,
    "combined": _utc(2025, 6, 5, 14, 0),
}
HARMONISED_END = {c: _utc(2025, 6, 5, 8, 0) for c in CONDITIONS}

# Training ran 26 May - 3 June 2025, with a break over the weekend of 31 May -
# 1 June. The final session ended ~13:34 BST on 3 June. Records after this
# boundary were collected with no further interaction.
TRAINING_END = _utc(2025, 6, 3, 14, 0)

# Video capture failed for all twelve accounts on these dates while Shorts and
# advertisement capture continued. Documented, not silently dropped.
OUTAGE_DATES = ["2025-05-30", "2025-05-31", "2025-06-01"]

# --- Model / analysis parameters -------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384 dimensions (the paper previously said 768)
EMBED_DIM = 384
PCA_COMPONENTS = 100
HDBSCAN_MIN_CLUSTER_SIZE = 50
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = 200

# Comparison sets differ in size and the best-match similarity measure grows
# with set size, so similarities are reported size-matched over this many draws.
N_SUBSAMPLE_DRAWS = 200

# GPT-4 labelling was run in June 2025. That model snapshot is no longer served,
# so labels ship as data (data/derived/llm_labels.csv) and are inputs here
# rather than regenerable outputs. See README.
LLM_MODEL_USED = "gpt-4 (June 2025 snapshot; alias, not pinned)"

# Cluster -> content category. Cluster numbering here is the raw HDBSCAN output
# in cluster_assignments.csv; the paper renumbers these 1-9 for presentation.
CLUSTER_TO_CATEGORY = {
    0: "mainstream",   # Relaxation and Study Music
    1: "puzzle",       # "The Witness"
    2: "generic",      # Roguelike and Tower Defense
    3: "puzzle",       # Mixed puzzle games (Death Squared 2, Talos, Gardens Between)
    4: "puzzle",       # "Lumines"
    5: "puzzle",       # Tetris
    6: "puzzle",       # "Puyo Puyo Tetris"
    7: "generic",      # "Top"/"Best" game reviews
    8: "generic",      # Super Nintendo World
}

LLM_TO_CATEGORY = {
    "gaming-puzzle": "puzzle",
    "gaming-other": "generic",
    "other": "mainstream",
}

CATEGORY_ORDER = ["mainstream", "generic", "puzzle"]
