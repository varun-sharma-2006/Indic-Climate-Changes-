# Dataset statistics

## English data by source, split and label

|                            |   Favour |   Against |   None |   Total |
|:---------------------------|---------:|----------:|-------:|--------:|
| ('ClimateStance', 'test')  |      285 |        35 |     35 |     355 |
| ('ClimateStance', 'train') |     2294 |       269 |    308 |    2871 |
| ('ClimateStance', 'val')   |      284 |        26 |     44 |     354 |
| ('GWSD', 'test')           |       76 |        39 |     85 |     200 |
| ('GWSD', 'train')          |      631 |       325 |    709 |    1665 |
| ('GWSD', 'val')            |       70 |        36 |     79 |     185 |
| ('Total', '')              |     3640 |       730 |   1260 |    5630 |

## Size of each split (per language; Hindi and Bengali are translations of the same rows)

| split   |   Favour |   Against |   None |   Total |
|:--------|---------:|----------:|-------:|--------:|
| train   |     2925 |       594 |   1017 |    4536 |
| val     |      354 |        62 |    123 |     539 |
| test    |      361 |        74 |    120 |     555 |
| Total   |     3640 |       730 |   1260 |    5630 |

## Class distribution (%)

| label   |   percent |
|:--------|----------:|
| Favour  |      64.7 |
| Against |      13   |
| None    |      22.4 |

## Text length in words (after cleaning)

|         |   mean |   50% |   max |
|:--------|-------:|------:|------:|
| Bengali |   21   |    18 |    94 |
| English |   24.8 |    23 |    78 |
| Hindi   |   28.3 |    25 |   108 |

## Average English length by source (words)

| source        |   mean words |
|:--------------|-------------:|
| ClimateStance |         28.3 |
| GWSD          |         18.8 |
