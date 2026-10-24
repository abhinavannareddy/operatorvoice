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
NEGATIVE = {"never", "ignores", "ignore", "wrong", "missing", "fails", "fail", "drops", "dropped", "lost",
            "cannot", "can't", "unreadable", "uncomfortable", "hurts", "useless", "loud", "noisy", "late",
            "too", "not", "no", "hard", "tiny", "poor", "miss", "waited", "frustrating", "anxiety", "mishear",
            "misunderstood", "forgets", "squeaks", "worn", "rattle", "optimistic", "mute", "distracting"}


def load_csv(path, text_col="text", rating_col="rating"):
    """Load real feedback. Only use posts you are allowed to analyse (check each site's terms)."""
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"column '{text_col}' not found; columns are {list(df.columns)}")
    out = pd.DataFrame({"text": df[text_col].astype(str)})
    out["rating"] = df[rating_col] if rating_col in df.columns else np.nan
    out["source"] = df["source"] if "source" in df.columns else "unknown"
    return out.dropna(subset=["text"]).reset_index(drop=True)


def clean(t: str) -> str:
    t = re.sub(r"https?://\S+", " ", t.lower())
    return re.sub(r"[^a-z0-9' ]+", " ", t)


def load_feasibility(path=None):
    cfg = yaml.safe_load(Path(path or ROOT / "config" / "feasibility.yaml").read_text())
    return set(cfg["software_terms"]), set(cfg["hardware_terms"])


def _contains_any(text: str, terms: set) -> bool:
    words = set(text.split())
    return any((t in words) if " " not in t else (t in text) for t in terms)


def cluster(texts, k=None, k_range=(6, 12), seed=0):
    small = len(texts) < 100            # tiny corpora cannot use document-frequency cut-offs
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1 if small else 5, max_df=1.0 if small else 0.3,
                          stop_words="english", sublinear_tf=True)
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
    df = df.copy()
    df["clean"] = df["text"].map(clean)
    labels, top_terms, k = cluster(df["clean"], k=k, seed=seed)
    df["cluster"] = labels
    sw, hw = load_feasibility(feasibility_path)
    neg = df["clean"].map(lambda t: min(1.0, sum(w in NEGATIVE for w in t.split()) / 2))
    low_rating = df["rating"].map(lambda r: 0.0 if pd.isna(r) else (1.0 if r <= 2 else 0.5 if r == 3 else 0.0))
    df["pain"] = np.where(df["rating"].isna(), neg, 0.7 * neg + 0.3 * low_rating)
    df["is_sw"] = df["clean"].map(lambda t: _contains_any(t, sw))
    df["is_hw"] = df["clean"].map(lambda t: _contains_any(t, hw))
    rows = []
    for c in range(k):
        g = df[df.cluster == c]
        feas = float(np.clip(g.is_sw.mean() - 0.8 * g.is_hw.mean(), 0, 1))
        rows.append({"cluster": c, "top_terms": ", ".join(top_terms[c]), "posts": len(g), "share_pct": round(100 * len(g) / len(df), 1),
                     "pain": round(g.pain.mean(), 3), "feasibility": round(feas, 3),
                     "example": g.sort_values("pain", ascending=False).text.iloc[0]})
    opp = pd.DataFrame(rows)
    opp["demand"] = opp.posts / opp.posts.max()
    opp["priority"] = (opp.demand * opp.pain * opp.feasibility).round(3)
    opp = opp.sort_values("priority", ascending=False).reset_index(drop=True)
    opp.insert(0, "rank", opp.index + 1)
    return {"posts": df, "opportunities": opp, "k": k}
