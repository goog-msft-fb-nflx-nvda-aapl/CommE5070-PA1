import os

PROJECT_ROOT = os.environ.get("PA1_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "extracted")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

SAMPLE_RATE = 24000
CLIP_SECONDS = 30.0
TARGET_LUFS = -23.0

DECADE_LABELS = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
DECADE_TO_INT = {lab: i for i, lab in enumerate(DECADE_LABELS)}

MARKET_LABELS = ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]
MARKET_TO_INT = {lab: i for i, lab in enumerate(MARKET_LABELS)}

DATASETS = {
    "A": {"dir": os.path.join(DATASET_DIR, "dataset_A"), "labels": DECADE_LABELS, "label_to_int": DECADE_TO_INT, "ordinal": True},
    "B": {"dir": os.path.join(DATASET_DIR, "dataset_B"), "labels": MARKET_LABELS, "label_to_int": MARKET_TO_INT, "ordinal": False},
}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
