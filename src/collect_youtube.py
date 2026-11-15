"""Collect Hindi / Bengali YouTube comments on climate videos (for our own labelled test set).

Needs a free YouTube Data API v3 key:  https://developers.google.com/youtube/v3/getting-started
    PowerShell:  $env:YT_API_KEY = "your-key"

Usage:  python src/collect_youtube.py --per_query_videos 10 --max_comments 100
Output: data/youtube/comments_to_label.csv  (fill the 'label' column with Favour / Against / None)

Only comment ID, video ID, text and a script-based language guess are stored (no usernames).
"""
import argparse
import json
import os
import re
import urllib.parse
import urllib.request

import pandas as pd

from config import DATA_DIR

API = "https://www.googleapis.com/youtube/v3/"
QUERIES = {
    "hi": ["जलवायु परिवर्तन", "ग्लोबल वार्मिंग", "जलवायु संकट भारत", "हीटवेव भारत गर्मी"],
    "bn": ["জলবায়ু পরিবর্তন", "বিশ্ব উষ্ণায়ন", "গ্লোবাল ওয়ার্মিং", "জলবায়ু সংকট"],
}


def call(endpoint, **params):
    params["key"] = os.environ["YT_API_KEY"]
    url = API + endpoint + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def guess_lang(text):
    deva = len(re.findall(r"[ऀ-ॿ]", text))
    beng = len(re.findall(r"[ঀ-৿]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if max(deva, beng, latin) == 0:
        return "other"
    return ["hi", "bn", "latin"][[deva, beng, latin].index(max(deva, beng, latin))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_query_videos", type=int, default=10)
    ap.add_argument("--max_comments", type=int, default=100, help="per video")
    args = ap.parse_args()
    if "YT_API_KEY" not in os.environ:
        raise SystemExit("Set the YT_API_KEY environment variable first (see the docstring).")

    rows = []
    for lang, queries in QUERIES.items():
        for q in queries:
            videos = call("search", part="id", q=q, type="video", maxResults=args.per_query_videos,
                          relevanceLanguage=lang)
            for item in videos.get("items", []):
                vid = item["id"]["videoId"]
                try:
                    res = call("commentThreads", part="snippet", videoId=vid, maxResults=args.max_comments,
                               textFormat="plainText", order="relevance")
                except Exception as e:  # comments disabled on some videos
                    print(f"skip {vid}: {e}")
                    continue
                for c in res.get("items", []):
                    s = c["snippet"]["topLevelComment"]["snippet"]
                    rows.append({"comment_id": c["id"], "video_id": vid, "query_lang": lang,
                                 "text": s["textDisplay"], "lang_guess": guess_lang(s["textDisplay"]), "label": ""})
            print(f"{q}: {len(rows)} comments so far")

    df = pd.DataFrame(rows).drop_duplicates("comment_id")
    df = df[df["text"].str.split().str.len() >= 3]
    out_dir = DATA_DIR / "youtube"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "comments_to_label.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} comments -> {out_dir / 'comments_to_label.csv'}")
    print(df["lang_guess"].value_counts())


if __name__ == "__main__":
    main()
