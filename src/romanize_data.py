"""Step 2b: romanised copies of the Hindi and Bengali data (data augmentation).

Many YouTube users type Hindi/Bengali in English letters ("garmi bahut badh gayi hai").
Our models scored only 20% on such comments in the challenge set, so we add romanised
versions of the translated data. We reuse romanize() from our Milestone 1 code
(stance/augment.py): ITRANS transliteration, lower-cased, with the final "a"
dropped (parivartana -> parivartan) so it looks like casual typing.

Output: data/processed/hindi_roman.csv, data/processed/bengali_roman.csv
Usage:  python src/romanize_data.py
"""
import sys

import pandas as pd

from config import PROCESSED_DIR, ROOT

sys.path.insert(0, str(ROOT))
from stance.augment import romanize  # noqa: E402

PAIRS = {"hindi.csv": "hindi_roman.csv", "bengali.csv": "bengali_roman.csv"}


def main():
    for src, dst in PAIRS.items():
        df = pd.read_csv(PROCESSED_DIR / src, keep_default_na=False)
        df["text"] = df["text"].map(romanize)
        df.to_csv(PROCESSED_DIR / dst, index=False)
        print(f"Saved {len(df)} rows -> {PROCESSED_DIR / dst}")
        print("   e.g.", df["text"].iloc[0][:100].encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
