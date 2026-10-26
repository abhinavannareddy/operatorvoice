"""Checks the pipeline against the planted themes of the synthetic corpus."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import synth
from .pipeline import analyse


def evaluate_one(seed: int) -> dict:
    df = synth.generate(seed)
    res = analyse(df, seed=0)
    posts, opp = res["posts"], res["opportunities"]
    truth = synth.ground_truth().set_index("theme")
    signal = posts[posts.theme != "none"]
    ari = adjusted_rand_score(signal.theme, signal.cluster)
    # match each planted theme to the cluster that holds most of its posts
    matched, purity_num = {}, 0
    for theme, g in signal.groupby("theme"):
        c = g.cluster.value_counts().idxmax()
        matched[theme] = c
        purity_num += int((g.cluster == c).sum())
    rec = []
    for theme, c in matched.items():
        row = opp[opp.cluster == c].iloc[0]
        rec.append(
            {
                "theme": theme,
                "true_priority": truth.loc[theme, "priority"],
                "pred_priority": row.priority,
                "true_feas": truth.loc[theme, "feasibility"],
                "pred_feas": row.feasibility,
            }
        )
    m = pd.DataFrame(rec)
    top5_true = set(m.nlargest(5, "true_priority").theme)
    top5_pred = set(m.nlargest(5, "pred_priority").theme)
    sw = m[m.true_feas >= 0.8].pred_feas.mean()
    hw = m[m.true_feas <= 0.1].pred_feas.mean()
    return {
        "seed": seed,
        "k": int(res["k"]),
        "ari": ari,
        "theme_recall": purity_num / len(signal),
        "priority_spearman": m.true_priority.corr(m.pred_priority, method="spearman"),
        "top5_overlap": len(top5_true & top5_pred) / 5,
        "sw_feasibility": sw,
        "hw_feasibility": hw,
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    df = pd.DataFrame([evaluate_one(s) for s in range(1, n + 1)])
    mean = df.drop(columns="seed").mean().round(3)
    lines = ["| Metric | Mean over seeds |", "|---|---|"]
    names = {
        "k": "clusters chosen",
        "ari": "adjusted Rand index vs planted themes",
        "theme_recall": "share of posts in their theme's main cluster",
        "priority_spearman": "Spearman: predicted vs planted priority",
        "top5_overlap": "overlap of top-5 opportunities with planted top-5",
        "sw_feasibility": "mean feasibility, software-addressable themes",
        "hw_feasibility": "mean feasibility, hardware themes",
    }
    for k, v in mean.items():
        lines.append(f"| {names[k]} | {v} |")
    text = "\n".join(lines)
    print(text)
    Path("results").mkdir(exist_ok=True)
    Path("results/latest.md").write_text(text + f"\n\n{n} seeds, 10 planted themes, about 530 posts each.\n")
    df.to_csv("results/per_seed.csv", index=False)


if __name__ == "__main__":
    main()
