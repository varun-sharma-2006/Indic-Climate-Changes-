# Our YouTube comment set (work in progress)

1. Get a free YouTube Data API v3 key and set it: `$env:YT_API_KEY = "..."`
2. Run `python src/collect_youtube.py` -> creates `comments_to_label.csv` here.
3. Two team members label each comment independently (`label` = Favour / Against / None),
   saved as `labels_annotator1.csv` and `labels_annotator2.csv`.
4. Disagreements are discussed; the final file is `youtube_labelled.csv` (columns: id, text, label, lang).

Labelling guide:
- Favour: says or implies climate change is real, harmful or needs action.
- Against: denies, mocks or downplays it.
- None: off-topic, praise for the video, questions without a stance, unclear.
Judge the stance toward the claim, not the sentiment (an angry comment about government inaction is Favour).
