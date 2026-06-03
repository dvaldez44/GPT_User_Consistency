from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

MODEL_NAME = "potsawee/deberta-v3-large-mnli"

USER_SOURCE = DATA_RAW / "exploded_data_filtered2.csv"
STAGE_SOURCES = {
    1: DATA_RAW / "exploded_data_stages1.csv",
    2: DATA_RAW / "exploded_data_stages2.csv",
    3: DATA_RAW / "exploded_data_stages3.csv",
}

PROMPT_LABELS = {
    0: "Diet",
    1: "Physical Activity",
    4: "Physical Activity",
}

USER_DOMAIN_LABELS = {
    "diet": "Diet",
    "pa": "Physical activity",
}

STAGE_PAIR_LABELS = {
    "1-2": "Pre-D/T2D",
    "1-3": "Pre-D/T2D+CVD",
    "1-4": "Pre-D/T2D+CKD",
    "1-5": "Pre-D/T2D+DR",
    "2-3": "T2D/T2D+CVD",
    "2-4": "T2D/T2D+CKD",
    "2-5": "T2D/T2D+DR",
    "3-4": "T2D+CVD/T2D+CKD",
    "3-5": "T2D+CVD/T2D+DR",
    "4-5": "T2D+CKD/T2D+DR",
}

STAGE_PAIR_ORDER = [
    "Pre-D/T2D",
    "Pre-D/T2D+CVD",
    "Pre-D/T2D+CKD",
    "Pre-D/T2D+DR",
    "T2D/T2D+CVD",
    "T2D/T2D+CKD",
    "T2D/T2D+DR",
    "T2D+CVD/T2D+CKD",
    "T2D+CVD/T2D+DR",
    "T2D+CKD/T2D+DR",
]
