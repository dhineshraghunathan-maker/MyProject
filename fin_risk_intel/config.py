"""Central configuration for the dissertation prototype."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, ART = ROOT / "data", ROOT / "artifacts"
IMG = DATA / "images"
for _p in (DATA, ART, IMG, ART / "model_cards"):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CUSTOMERS = 3000                 # synthetic, de-identified customers
REVIEW_CAPACITY = 0.05             # share of test-period customers investigators can review
ESCALATE_SHARE = 0.015             # top share of test customers routed to ESCALATE
FAIRNESS_GAP_THRESHOLD = 0.10      # agreed maximum FPR / FNR gap between segments (H6)
RETRIEVAL_CONFIDENCE_MIN = 0.30    # below this the copilot declines a definitive conclusion
USE_TRANSFORMERS = os.getenv("USE_TRANSFORMERS", "0") == "1"   # real dense + cross-encoder models
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")         # used only if ANTHROPIC_API_KEY is set

# Time-and-motion assumptions for H5 (SIMULATION - replace with measured pilot data)
HANDLING = dict(manual_base=6.0, per_source=4.5, n_sources=5, assisted_base=3.0, verify_per_citation=0.8)
