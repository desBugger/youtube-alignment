# Two Buttons, One Feed

Data and analysis code for a controlled sock-puppet audit of YouTube's explicit
feedback controls (`like` and `not interested`), run as a 2×2 factorial design
across twelve simulated accounts.

**Headline finding.** No single signal aligned the feed. Accounts that watched
seventy on-interest videos to completion, engaging with nothing else, still
received on-interest recommendations one time in ten. Liking raised that to
roughly one and a half; declining unwanted content to two. Only using **both**
controls reached four in ten — and the two combine super-additively, so their
value depends on being used together.

## Quick start

```bash
pip install -r requirements.txt
python src/pipeline.py
```

This regenerates every number reported in the paper and writes tables to
`results/tables/`. It runs in under a minute on a laptop; no GPU required.

To use a single common observation window across all four conditions instead of
the per-condition cutoffs used in the paper:

```bash
python src/pipeline.py --window harmonised
```

## The experiment

Twelve YouTube accounts configured as 13-year-olds were trained in synchrony
for seven daily sessions (26 May – 3 June 2025, with a weekend break), each
watching ten puzzle-gaming videos per day in full. Accounts were split across
four conditions, three each:

| condition | watches | likes | marks "not interested" |
|---|---|---|---|
| `watch` (control) | ✓ | | |
| `like` | ✓ | ✓ | |
| `notInt` | ✓ | | ✓ |
| `combined` | ✓ | ✓ | ✓ |

Homepage recommendations were captured hourly. Device, browser, geolocation and
account settings were held fixed; all twelve sessions ran in parallel to
minimise temporal drift.

## Repository layout

```
data/raw/          hourly homepage captures, one row per item per account
data/derived/      embeddings, cluster assignments, LLM category labels,
                   and the assembled analysed corpus
src/config.py      every parameter, window and seed
src/pipeline.py    raw captures -> reported numbers
results/tables/    generated output
docs/DATASHEET.md  datasheet for the dataset (Gebru et al., 2021)
```

## Data

`data/raw/feedwatch{6,7}.csv` — one row per distinct item observed per account,
with `count`, `persona`, `type`, `title`, `url`, `channel`, `channel_url`,
`firstSeen`, `lastSeen`. Four item types are recorded:

| type | rows | distinct |
|---|---|---|
| `video` — standard homepage recommendations | 7,716 | — |
| `short` — Shorts shelf | 4,349 | 1,398 |
| `feedAd` — in-feed advertisements | 1,447 | 103 |
| `videoAd` — in-video advertisements | 717 | 75 |

Only `video` rows are analysed in the paper. The Shorts and advertisement
records are released and not analysed.

Every account was created by the researchers.

## Known data issues

These are documented in the paper and reproduced by the pipeline.

1. **Video capture outage, 30 May – 1 June 2025.** A parser fault suspended
   capture of standard video recommendations while Shorts and advertisement
   capture continued normally. It affected **all twelve accounts identically**,
   so it reduces observation time without biasing between-condition comparison.
   `python src/pipeline.py` prints the per-condition, per-date counts showing it.

2. **One account failed.** `dislike3gamer2012` returned no video
   recommendations after 28 May, contributing 40 of 1,346 not-interested
   records (3.0%). The paper reports that condition as n=2. Set
   `EXCLUDE_FAILED_ACCOUNT = True` in `config.py` to drop it entirely as a
   sensitivity check.

3. **Differing per-condition end cutoffs.** The original analysis applied a
   different end cutoff to each condition and none at all to `notInt`, giving
   that condition roughly 18 hours more observation. `--window published`
   reproduces this; `--window harmonised` applies one common end to all four.

4. **Uneven Shorts volume across conditions.** Shorts counts differ markedly
   between conditions. This may be a real difference in how the Shorts shelf
   responds to explicit feedback, or a collection artefact. We flag it as an
   open question rather than a finding.

## Licences

- Code: MIT (`LICENSE`)
- Data: CC BY 4.0 (`LICENSE-DATA`)

Dependencies: `sentence-transformers` (Apache 2.0), `all-MiniLM-L6-v2`
(Apache 2.0), `hdbscan` (BSD-3), `scikit-learn` (BSD-3), `langdetect`
(Apache 2.0). Collection tool adapted from
[TheirTube](https://www.tomokihara.com/en/theirtube.html) (Kihara, 2020).

## Ethics

No human subjects. All twelve accounts were created by the researchers and the
records are public video metadata served to those accounts. The study used a
small number of accounts over one week on age-appropriate content to limit any
effect on recommendations served to real users. Institutional ethics approval
was obtained.

## Citation

If you use this dataset or code, please cite the paper:

```bibtex
@misc{cho2026twobuttons,
  title        = {Two Buttons, One Feed: Aligning YouTube Recommendations Requires Both Controls},
  author       = {Cho, Desiree and Hale, Scott A. and Zhao, Jun and Shadbolt, Nigel and Przybylski, Andrew K.},
  year         = {2026},
  note         = {Preprint; under review}
}
```

Machine-readable metadata is in `CITATION.cff`. This entry will be replaced with
the published reference if the paper is accepted.
