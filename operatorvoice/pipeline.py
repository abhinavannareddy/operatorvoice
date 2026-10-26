"""Groups unmet needs in free-text feedback and ranks software-feature opportunities."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).resolve().parent.parent
NEGATIVE = {
    "never",
    "ignores",
    "ignore",
    "wrong",
    "missing",
    "fails",
    "fail",
    "drops",
    "dropped",
    "lost",
    "cannot",
    "can't",
    "unreadable",
    "uncomfortable",
    "hurts",
    "useless",
    "loud",
    "noisy",
    "late",
    "too",
    "not",
    "no",
    "hard",
    "tiny",
    "poor",
    "miss",
    "waited",
    "frustrating",
    "anxiety",
    "mishear",
    "misunderstood",
    "forgets",
    "squeaks",
    "worn",
    "rattle",
    "optimistic",
    "mute",
    "distracting",
}


MAX_ROWS = 200_000  # guard against accidentally loading something enormous into memory
MIN_POSTS = 12  # below this, clustering into 6+ groups is meaningless


def load_csv(path, text_col="text", rating_col="rating", max_rows=MAX_ROWS):
    """Load real feedback. Only use posts you are allowed to analyse (check each site's terms).

    Rows with empty text are dropped. Ratings outside 1-5 are rejected rather than silently clipped,
    because a wrong scale (for example 0-10) would distort the pain score without any visible error."""
    df = pd.read_csv(path, nrows=max_rows + 1)
    if text_col not in df.columns:
        raise ValueError(f"column '{text_col}' not found; columns are {list(df.columns)}")
    if len(df) > max_rows:
        raise ValueError(f"more than {max_rows} rows; split the file or raise max_rows")
    out = pd.DataFrame({"text": df[text_col]})
    out["rating"] = pd.to_numeric(df[rating_col], errors="coerce") if rating_col in df.columns else np.nan
    out["source"] = df["source"].fillna("unknown") if "source" in df.columns else "unknown"
    out = out.dropna(subset=["text"])
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""].reset_index(drop=True)
    if ((out["rating"] < 1) | (out["rating"] > 5)).any():
        raise ValueError("ratings must be between 1 and 5 (leave the cell empty when there is no rating)")
    return out


def clean(t: str) -> str:
    t = re.sub(r"https?://\S+", " ", t.lower())
    return re.sub(r"[^a-z0-9' ]+", " ", t)


def load_feasibility(path=None):
    cfg = yaml.safe_load(Path(path or ROOT / "config" / "feasibility.yaml").read_text())
    return set(cfg["software_terms"]), set(cfg["hardware_terms"])


def _matches(text: str, terms: set) -> list:
    """Terms from `terms` found in cleaned text. Kept as a list so the report can show the evidence."""
    words = set(text.split())
    return sorted(t for t in terms if ((t in words) if " " not in t else (t in text)))


def cluster(texts, k=None, k_range=(6, 12), seed=0):
    small = len(texts) < 100  # tiny corpora cannot use document-frequency cut-offs
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1 if small else 5,
        max_df=1.0 if small else 0.3,
        stop_words="english",
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    comps = max(1, min(25, X.shape[1] - 1, X.shape[0] - 1))
    Z = Normalizer().fit_transform(TruncatedSVD(comps, random_state=seed).fit_transform(X))
    if k is None:
        best, k = -1, k_range[0]
        for kk in range(k_range[0], min(k_range[1], len(texts) - 1) + 1):
            labels = KMeans(kk, n_init=5, random_state=seed).fit_predict(Z)
            if len(set(labels)) < 2:
                continue
            s = silhouette_score(Z, labels)
            if s > best:
                best, k = s, kk
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    terms = np.array(vec.get_feature_names_out())
    centroids = np.vstack([np.asarray(X[km.labels_ == c].mean(axis=0)).ravel() for c in range(k)])
    top = [list(terms[np.argsort(-row)[:5]]) for row in centroids]
    return km.labels_, top, k


def analyse(df: pd.DataFrame, k=None, feasibility_path=None, seed=0) -> dict:
    if "text" not in df.columns:
        raise ValueError("input needs a 'text' column")
    if k is None and len(df) < MIN_POSTS:  # automatic k searches 6-12 groups; an explicit k is the caller's call
        raise ValueError(
            f"need at least {MIN_POSTS} posts to choose the cluster count, got {len(df)}; pass k to override"
        )
    df = df.copy()
    df["clean"] = df["text"].map(clean)
    labels, top_terms, k = cluster(df["clean"], k=k, seed=seed)
    df["cluster"] = labels
    sw, hw = load_feasibility(feasibility_path)
    neg = df["clean"].map(lambda t: min(1.0, sum(w in NEGATIVE for w in t.split()) / 2))
    low_rating = df["rating"].map(lambda r: 0.0 if pd.isna(r) else (1.0 if r <= 2 else 0.5 if r == 3 else 0.0))
    df["pain"] = np.where(df["rating"].isna(), neg, 0.7 * neg + 0.3 * low_rating)
    df["sw_terms"] = df["clean"].map(lambda t: _matches(t, sw))
    df["hw_terms"] = df["clean"].map(lambda t: _matches(t, hw))
    df["neg_terms"] = df["clean"].map(lambda t: sorted({w for w in t.split() if w in NEGATIVE}))
    df["is_sw"] = df["sw_terms"].map(bool)
    df["is_hw"] = df["hw_terms"].map(bool)
    rows = []
    for c in range(k):
        g = df[df.cluster == c]
        feas = float(np.clip(g.is_sw.mean() - 0.8 * g.is_hw.mean(), 0, 1))
        rows.append(
            {
                "cluster": c,
                "top_terms": ", ".join(top_terms[c]),
                "posts": len(g),
                "share_pct": round(100 * len(g) / len(df), 1),
                "pain": round(g.pain.mean(), 3),
                "feasibility": round(feas, 3),
                "example": g.sort_values("pain", ascending=False).text.iloc[0],
            }
        )
    opp = pd.DataFrame(rows)
    opp["demand"] = opp.posts / opp.posts.max()
    opp["priority"] = (opp.demand * opp.pain * opp.feasibility).round(3)
    opp = opp.sort_values("priority", ascending=False).reset_index(drop=True)
    opp.insert(0, "rank", opp.index + 1)
    return {"posts": df, "opportunities": opp, "k": k}


def priority_score(demand, pain, feasibility, weights=(1.0, 1.0, 1.0)):
    """demand^a x pain^b x feasibility^c. The default exponents of 1 give the original formula.
    A product means a zero on any factor removes the opportunity, which is intended: a need nobody
    can fix in software should not rank however many people complain."""
    a, b, c = weights
    return demand**a * pain**b * feasibility**c


def bootstrap_priority(result: dict, n_boot: int = 300, seed: int = 0, top_n: int = 3) -> pd.DataFrame:
    """Sampling uncertainty of each cluster's priority.

    Clusters are held fixed and posts are resampled inside each cluster, so this measures how much the
    ranking depends on *which* posts were collected. It does not measure uncertainty from the clustering
    itself (a different k or seed can move cluster edges)."""
    posts, opp = result["posts"], result["opportunities"]
    rng = np.random.default_rng(seed)
    groups = {c: g[["pain", "is_sw", "is_hw"]].to_numpy(float) for c, g in posts.groupby("cluster")}
    clusters = opp["cluster"].tolist()
    draws = np.zeros((n_boot, len(clusters)))
    for b in range(n_boot):
        stats = []
        for c in clusters:
            g = groups[c]
            idx = rng.integers(0, len(g), len(g))  # same cluster size, so demand is held fixed
            m = g[idx].mean(axis=0)
            stats.append((len(g), m[0], float(np.clip(m[1] - 0.8 * m[2], 0, 1))))
        sizes = np.array([s[0] for s in stats], float)
        draws[b] = [priority_score(n / sizes.max(), p, f) for n, p, f in stats]
    ranks = (-draws).argsort(axis=1).argsort(axis=1) + 1
    out = opp[["rank", "cluster"]].copy()
    out["priority_lo"] = np.percentile(draws, 5, axis=0).round(3)
    out["priority_hi"] = np.percentile(draws, 95, axis=0).round(3)
    out["p_top_n"] = (ranks <= top_n).mean(axis=0).round(2)
    return out
