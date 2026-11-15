"""Global constants: paths, label set, claim text in each language."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OFFICIAL = DATA / "official"
EXT_RAW = DATA / "external" / "raw"
EXT_PROC = DATA / "external" / "processed"
TRANSLATED = DATA / "translated"
PROCESSED = DATA / "processed"
CHALLENGE = DATA / "challenge"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RUNS = ROOT / "runs"

# Canonical label order. Index = class id used by every model in this repo.
LABELS = ["Favour", "Against", "None"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# The target claim, given to target-aware models as the second segment.
CLAIM = {
    "en": "Climate change and global warming is a serious concern.",
    "hi": "जलवायु परिवर्तन और ग्लोबल वार्मिंग एक गंभीर चिंता का विषय है।",
    "bn": "জলবায়ু পরিবর্তন এবং বিশ্ব উষ্ণায়ন একটি গুরুতর উদ্বেগের বিষয়।",
}

# Unified schema every dataset is converted to.
COLUMNS = ["id", "text", "label", "lang", "source"]

SEED = 42
