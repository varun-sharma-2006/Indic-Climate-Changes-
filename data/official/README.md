Put the files you received by email here (any of .csv / .tsv / .xlsx / .json), e.g. `train.csv`, `test.csv`.
Then convert them:

    python -m stance.prepare_official --file data/official/train.csv --split train
    python -m stance.prepare_official --file data/official/test.csv  --split test

These files are git-ignored by default (check the organisers' redistribution terms before publishing).
