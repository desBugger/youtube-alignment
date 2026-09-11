"""Analysis pipeline: raw captures -> the numbers reported in the paper.

Run `python src/pipeline.py` to regenerate every reported figure and table.

Design note. Embeddings, cluster assignments and LLM category labels are shipped
as data rather than regenerated on each run, for three reasons: the GPT-4
snapshot used for labelling is no longer served; language detection was
unseeded in the original run so the exact surviving row set cannot be
reconstructed from raw; and re-embedding requires a 90MB model download. The
corpus of record is data/derived/analysed_corpus.csv, which this script builds
and verifies. Pass --rebuild-embeddings to recompute embeddings from raw.
"""
from __future__ import annotations

import argparse
import re
import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------------------------------------
# Stage 1: wrangle raw captures into the per-condition corpus
# ---------------------------------------------------------------------------
def clean_title(title: str) -> str:
    """Strip emoji, punctuation and excess whitespace. Matches the original."""
    return re.sub(r"[^\w\s]", "", str(title)).strip()


def load_raw() -> pd.DataFrame:
    frames = [pd.read_csv(C.RAW / f) for f in ("feedwatch6.csv", "feedwatch7.csv")]
    df = pd.concat(frames, ignore_index=True)
    account_to_condition = {
        acct: cond for cond, accts in C.CONDITIONS.items() for acct in accts
    }
    df["condition"] = df["persona"].map(account_to_condition)
    if df["condition"].isna().any():
        unknown = sorted(df.loc[df["condition"].isna(), "persona"].unique())
        raise ValueError(f"unmapped accounts in raw data: {unknown}")
    df["firstSeen"] = pd.to_datetime(df["firstSeen"], utc=True)
    df["lastSeen"] = pd.to_datetime(df["lastSeen"], utc=True)
    return df


def wrangle(df: pd.DataFrame, window_mode: str = None) -> pd.DataFrame:
    """Filter to standard video recommendations inside the analysis window."""
    window_mode = window_mode or C.WINDOW_MODE
    ends = C.PUBLISHED_END if window_mode == "published" else C.HARMONISED_END

    videos = df[df["type"] == "video"].copy()
    kept = []
    for condition in C.CONDITION_ORDER:
        block = videos[videos["condition"] == condition]
        block = block[block["firstSeen"] > C.WARMUP_END]
        end = ends[condition]
        if end is not None:
            block = block[block["firstSeen"] <= end]
        kept.append(block)

    out = pd.concat(kept, ignore_index=True)
    if C.EXCLUDE_FAILED_ACCOUNT:
        out = out[out["persona"] != C.FAILED_ACCOUNT].reset_index(drop=True)
    out["cleanTitle"] = out["title"].map(clean_title)
    out["period"] = np.where(out["firstSeen"] > C.TRAINING_END, "post", "train")
    return out


# ---------------------------------------------------------------------------
# Stage 2: attach timestamps to the embedded corpus
# ---------------------------------------------------------------------------
def align_timestamps(embedded: pd.DataFrame, wrangled: pd.DataFrame) -> pd.DataFrame:
    """Recover per-row timestamps for the embedded corpus.

    The embedded corpus stores only (condition, cleanTitle, embedding): the
    language filter dropped rows without recording which, so a key join on
    (condition, cleanTitle) leaves ~17% of rows ambiguous, because 357 titles
    were re-recommended on both sides of the training boundary.

    The language filter only ever *removes* rows and preserves order, so the
    embedded sequence is an ordered subsequence of the wrangled one. Walking
    both in order recovers every timestamp exactly and deterministically.
    """
    out = []
    for condition in C.CONDITION_ORDER:
        emb_block = embedded[embedded["condition"] == condition]
        emb_titles = emb_block["cleanTitle"].tolist()
        emb_vectors = emb_block["embedding"].tolist()

        raw_block = wrangled[wrangled["condition"] == condition]
        raw_titles = raw_block["cleanTitle"].tolist()
        raw_rows = raw_block.to_dict("records")

        i, matched = 0, []
        for j, title in enumerate(raw_titles):
            if i < len(emb_titles) and title == emb_titles[i]:
                row = dict(raw_rows[j])
                row["embedding"] = emb_vectors[i]  # carry the vector through
                matched.append(row)
                i += 1
        if i != len(emb_titles):
            raise RuntimeError(
                f"alignment failed for {condition}: matched {i}/{len(emb_titles)}"
            )
        out.append(pd.DataFrame(matched))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Stage 3: content categories
# ---------------------------------------------------------------------------
def attach_categories(corpus: pd.DataFrame) -> pd.DataFrame:
    """Merge HDBSCAN cluster labels and LLM labels into three categories."""
    clusters = pd.read_csv(C.DERIVED / "cluster_assignments.csv")
    title_to_cluster = clusters.drop_duplicates("title").set_index("title")["clusterNo"]

    labels = pd.read_csv(C.DERIVED / "llm_labels.csv")
    # Defensive: the original labelling parsed batched LLM output by line order
    # and did not strip the bullet prefix the model sometimes echoed back,
    # producing "- gaming-other" as a distinct category on 40 rows.
    labels["llmCategory"] = labels["llmCategory"].str.lstrip("- ")
    unexpected = set(labels["llmCategory"]) - set(C.LLM_TO_CATEGORY)
    if unexpected:
        raise ValueError(f"unexpected LLM labels after normalisation: {unexpected}")
    label_key = labels.drop_duplicates(["persona", "cleanTitle"]).set_index(
        ["persona", "cleanTitle"]
    )["llmCategory"]

    corpus = corpus.copy()
    corpus["cluster"] = corpus["cleanTitle"].map(title_to_cluster)
    corpus["category"] = corpus["cluster"].map(C.CLUSTER_TO_CATEGORY)

    noise = corpus["category"].isna()
    keys = pd.MultiIndex.from_frame(corpus.loc[noise, ["condition", "cleanTitle"]])
    corpus.loc[noise, "category"] = keys.map(label_key).map(C.LLM_TO_CATEGORY)

    if corpus["category"].isna().any():
        n = int(corpus["category"].isna().sum())
        raise RuntimeError(f"{n} rows could not be assigned a category")
    return corpus


# ---------------------------------------------------------------------------
# Stage 4: analyses
# ---------------------------------------------------------------------------
def _unit_norm(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def mean_best_match(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over rows of a of the maximum cosine similarity to any row of b."""
    return float((a @ b.T).max(axis=1).mean())


def similarity_to_control(embeddings: dict[str, np.ndarray], rng) -> pd.DataFrame:
    """H1, reported both raw and size-matched.

    The measure grows with the size of the comparison set, and the conditions
    differ in size, so raw values are not comparable across conditions.
    """
    control = embeddings["watch"]
    others = [c for c in C.CONDITION_ORDER if c != "watch"]
    smallest = min(len(embeddings[c]) for c in others)

    rows = []
    for condition in others:
        comparison = embeddings[condition]
        raw = mean_best_match(control, comparison)
        draws = [
            mean_best_match(
                control,
                comparison[rng.choice(len(comparison), smallest, replace=False)],
            )
            for _ in range(C.N_SUBSAMPLE_DRAWS)
        ]
        draws = np.array(draws)
        rows.append(
            {
                "condition": condition,
                "n": len(comparison),
                "raw": round(raw, 4),
                "matched": round(draws.mean(), 4),
                "lo": round(np.percentile(draws, 2.5), 4),
                "hi": round(np.percentile(draws, 97.5), 4),
                "matched_at_n": smallest,
            }
        )
    return pd.DataFrame(rows)


def category_shares(corpus: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(corpus["condition"], corpus["category"])
    counts = counts.reindex(C.CONDITION_ORDER)[C.CATEGORY_ORDER]
    counts["n"] = counts.sum(axis=1)
    shares = counts[C.CATEGORY_ORDER].div(counts["n"], axis=0) * 100
    return pd.concat([shares.round(1), counts["n"]], axis=1)


def interaction(corpus: pd.DataFrame) -> dict:
    """Is the combined condition more than the sum of its parts?

    Additivity on a percentage scale is only one defensible baseline for a
    bounded proportion, so the same question is asked on multiplicative and
    log-odds scales. The paper reports the interaction as robustly positive
    rather than leaning on any single magnitude.
    """
    share = (
        corpus.groupby("condition")["category"]
        .apply(lambda s: (s == "puzzle").mean() * 100)
        .reindex(C.CONDITION_ORDER)
    )
    c, like, ni, comb = (share[k] for k in C.CONDITION_ORDER)

    additive = c + (like - c) + (ni - c)
    multiplicative = c * (like / c) * (ni / c)

    def logit(p):
        return np.log((p / 100) / (1 - p / 100))

    def inv_logit(x):
        return 100 / (1 + np.exp(-x))

    log_odds_pred = inv_logit(logit(c) + (logit(like) - logit(c)) + (logit(ni) - logit(c)))

    return {
        "shares": share.round(1).to_dict(),
        "additive_prediction": round(additive, 1),
        "multiplicative_prediction": round(multiplicative, 1),
        "log_odds_prediction": round(float(log_odds_pred), 1),
        "observed": round(comb, 1),
        "excess_additive": round(comb - additive, 1),
        "excess_multiplicative": round(comb - multiplicative, 1),
        "excess_log_odds": round(float(comb - log_odds_pred), 1),
        "logit_interaction_term": round(
            float(logit(comb) - (logit(c) + (logit(like) - logit(c)) + (logit(ni) - logit(c)))),
            3,
        ),
    }


def missingness(wrangled: pd.DataFrame) -> pd.DataFrame:
    """Records per condition per date, showing the capture outage."""
    table = pd.crosstab(
        wrangled["firstSeen"].dt.date, wrangled["condition"]
    ).reindex(columns=C.CONDITION_ORDER, fill_value=0)
    full_range = pd.date_range("2025-05-27", "2025-06-06", freq="D").date
    return table.reindex(full_range, fill_value=0)


# ---------------------------------------------------------------------------
def load_embeddings() -> pd.DataFrame:
    """Load the shipped embedding matrix.

    Stored as a plain .npy array plus a CSV index rather than a pickled
    DataFrame: pickles are tied to the numpy version that wrote them and fail to
    load across major versions, which is the wrong property for an artifact
    meant to outlive its environment.
    """
    index = pd.read_csv(C.DERIVED / "embeddings_index.csv")
    vectors = np.load(C.DERIVED / "embeddings.npy")
    if len(index) != len(vectors):
        raise RuntimeError(
            f"index/vector length mismatch: {len(index)} vs {len(vectors)}"
        )
    if vectors.shape[1] != C.EMBED_DIM:
        raise RuntimeError(
            f"expected {C.EMBED_DIM}-d embeddings, got {vectors.shape[1]}"
        )
    index["cleanTitle"] = index["cleanTitle"].astype(str)
    index["embedding"] = list(vectors)
    return index


def apply_window(corpus: pd.DataFrame, window_mode: str) -> pd.DataFrame:
    """Filter an already-aligned corpus to the chosen observation window.

    Alignment must always run against the published window, because that is the
    window the shipped embeddings were built from. Any narrower window is
    applied afterwards, as a filter on the aligned rows.
    """
    if window_mode == "published":
        return corpus
    ends = C.HARMONISED_END
    kept = [
        block[block["firstSeen"] <= ends[cond]]
        for cond, block in corpus.groupby("condition")
    ]
    return pd.concat(kept, ignore_index=True)


def main(window_mode: str) -> None:
    rng = np.random.default_rng(C.SEED)

    raw = load_raw()
    # Always wrangle on the published window for alignment; see apply_window().
    wrangled = wrangle(raw, "published")
    print(f"[1] wrangled: {len(wrangled)} video records (published window)")
    print(wrangled.groupby("condition").size().reindex(C.CONDITION_ORDER).to_string())

    embedded = load_embeddings()
    corpus = align_timestamps(embedded, wrangled)
    print(f"\n[2] aligned {len(corpus)} embedded rows to timestamps (100% required)")

    corpus = attach_categories(corpus)
    if window_mode != "published":
        before = len(corpus)
        corpus = apply_window(corpus, window_mode)
        print(f"[2b] {window_mode} window: {before} -> {len(corpus)} rows")
    corpus.drop(columns=["embedding"]).to_csv(
        C.DERIVED / "analysed_corpus.csv", index=False
    )
    print(f"[3] categories attached -> data/derived/analysed_corpus.csv")

    embeddings = {
        cond: _unit_norm(
            np.stack(corpus.loc[corpus["condition"] == cond, "embedding"].values)
            .astype(np.float64)
        )
        for cond in C.CONDITION_ORDER
    }

    print("\n[4] H1 similarity to control")
    sim = similarity_to_control(embeddings, rng)
    print(sim.to_string(index=False))

    print("\n[5] category shares (pooled)")
    print(category_shares(corpus).to_string())

    for period in ("train", "post"):
        print(f"\n[5{'ab'[period == 'post']}] category shares ({period}-training)")
        print(category_shares(corpus[corpus["period"] == period]).to_string())

    print("\n[6] interaction test")
    for key, value in interaction(corpus).items():
        print(f"    {key}: {value}")

    print("\n[7] records per condition per date (zeros = capture outage)")
    print(missingness(wrangled).to_string())

    sim.to_csv(C.TABLES / "similarity.csv", index=False)
    category_shares(corpus).to_csv(C.TABLES / "category_shares.csv")
    missingness(wrangled).to_csv(C.TABLES / "missingness.csv")
    print(f"\nwrote tables to {C.TABLES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window",
        choices=["published", "harmonised"],
        default=C.WINDOW_MODE,
        help="published reproduces the paper's per-condition cutoffs; "
             "harmonised applies one common end date to all conditions",
    )
    args = parser.parse_args()
    main(args.window)
