# Datasheet: YouTube Feedback-Control Audit Corpus

Following Gebru et al. (2021), *Datasheets for Datasets*.

## Motivation

**Why was the dataset created?** To measure whether YouTube's explicit feedback
controls (`like`, `not interested`) let a user steer recommendations toward a
stated interest, and whether the two controls are equally effective. Existing
audits test whether users can *remove* unwanted content; none had crossed both
controls factorially, and the positive control was largely untested.

**Who created it?** The paper's authors, at a university research institute.
Funded by public research funding and philanthropic foundations (details in the
camera-ready acknowledgements).

## Composition

**What do instances represent?** One row per distinct item observed on the
YouTube homepage of one account, with first and last observation timestamps.

**Fields.** `count` (times observed), `persona` (account identifier),
`type`, `title`, `url`, `channel`, `channel_url`, `firstSeen`, `lastSeen`.

**How many instances?**

| type | rows | distinct titles |
|---|---|---|
| `video` (standard recommendations) | 7,716 | — |
| `short` (Shorts shelf) | 4,349 | 1,398 |
| `feedAd` (in-feed advertisements) | 1,447 | 103 |
| `videoAd` (in-video advertisements) | 717 | 75 |

The paper analyses 5,296 `video` rows after windowing and language filtering.
Shorts and advertisement records are released unanalysed.

**Does it contain personal data?** No. All twelve accounts were created by the
researchers for this study. Account identifiers are researcher-assigned
pseudonyms, not real users. All content records are public video metadata.

**Does it contain offensive content?** Titles were captured as served by the
platform, on accounts trained exclusively on age-appropriate puzzle-gaming
content. We have not exhaustively reviewed every title and do not warrant that
none is objectionable.

**Is anything missing?** Yes, and deliberately documented rather than patched:

- A parser fault suspended `video` capture for **all twelve accounts** from
  30 May to 1 June 2025, while Shorts and advertisement capture continued. The
  outage is common to all conditions.
- `dislike3gamer2012` returned no `video` records after 28 May, contributing 40
  rows (3.0% of its condition). The paper reports that condition as n=2.
- The original analysis applied different end cutoffs per condition, and none to
  `notInt`. Both the original and a harmonised window are supported in code.

**Is it self-contained?** Yes. Video URLs point to YouTube and may rot; titles,
channels and timestamps are captured in full and do not depend on the platform.

## Collection

**How was it collected?** An automated scraper, adapted from TheirTube (Kihara,
2020), captured logged-in homepage recommendations hourly. The YouTube API does
not expose homepage recommendations, so browser-based collection was necessary.

**Over what period?** 26 May – 6 June 2025. Training sessions ran 26 May –
3 June with a weekend break; collection continued three days past the final
session, which the paper uses to test persistence.

**Sampling.** Not a sample. It is the complete capture for twelve accounts in a
2×2 factorial design (three replicates per cell) crossing `like` and
`not interested`, with device, browser, geolocation and account settings held
fixed and all sessions run in parallel.

**Who collected it?** The authors. Feedback actions were applied manually to
ensure consistent judgement on ambiguous cases; capture was automated.

**Ethics.** No human subjects; no consent was required or sought, as no personal
data were collected. Institutional ethics approval was obtained. Scale was kept
small (12 accounts, one week, age-appropriate content) to limit any effect on
recommendations served to real users.

## Preprocessing

Analysis retains `video` rows inside the analysis window, drops non-English
titles, and strips emoji, punctuation and excess whitespace. **The raw captures
are released unmodified**; all filtering happens in `src/pipeline.py` and is
reversible.

Derived artifacts shipped alongside: sentence embeddings
(`all-MiniLM-L6-v2`, 384-d), HDBSCAN cluster assignments, and GPT-4 category
labels. The labels ship as data because the model snapshot used is no longer
served; see README.

## Uses

**Used for.** The accompanying paper: divergence from control, content-category
composition, the interaction between the two controls, and a period split.

**Other uses we did not pursue.** Whether the Shorts shelf responds to the same
feedback signals as the main feed; whether advertisement targeting shifts with
expressed interests; temporal dynamics at hourly resolution; channel-level
concentration.

**Uses to avoid.** The data reflect one platform, one interest, one age profile,
and one week in 2025. YouTube's recommender changes continuously and runs
concurrent experiments. Absolute values should not be treated as stable platform
properties; the comparative structure between conditions is what the design
supports. The data cannot support claims about real users' behaviour, since no
real users were observed.

## Distribution

Public, under CC BY 4.0, via GitHub, Hugging Face, and an archival deposit with
a persistent identifier.

## Maintenance

Maintained by the authors. This is a fixed observational record of a
one-week experiment; it will not be extended. Corrections, if needed, will be
issued as new versions with the original retained.
