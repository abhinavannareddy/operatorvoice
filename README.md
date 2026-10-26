# OperatorVoice

![ci](https://github.com/abhinavannareddy/operatorvoice/actions/workflows/ci.yml/badge.svg)

OperatorVoice reads free-text feedback from drivers, operators and car owners, groups the unmet needs
people describe, and ranks which ones a software team could actually address. The output is a ranked table
(CSV, SQLite) and a self-contained HTML report where you can change the weighting and see the evidence
behind every score.

## Problem

Feedback arrives as thousands of unstructured posts. Counting complaints favours loud topics such as
seat comfort or road noise, which software cannot fix. A useful ranking has to combine how many people
raise a need, how much it hurts, and whether software can plausibly address it, and it has to show its
working so a product owner can disagree with it.

## Approach

```
CSV ─► clean ─► TF-IDF (1-2 grams) ─► SVD ─► k-means (k by silhouette) ─► clusters
                                                                            │
 config/feasibility.yaml ─► software / hardware term matching ─► per-cluster demand, pain, feasibility
                                                                            │
              priority = demand x pain x feasibility ─► CSV + SQLite + report.html
```

Full details, including formulas and trade-offs, are in [docs/architecture.md](docs/architecture.md).

## Capabilities

- Cluster feedback and label each cluster with its top terms.
- Score demand, pain (negative wording plus rating when present) and feasibility (software vs hardware terms from an editable YAML).
- Per-post evidence: the exact words that raised its pain and feasibility scores.
- Bootstrap range for each priority and the chance a cluster is in the top three.
- HTML report: re-weight the three factors with sliders, re-sort live, drill into a cluster's posts. Works offline, keyboard accessible, follows the system dark/light setting.
- Synthetic corpus with 10 planted themes, and an evaluation script that checks the pipeline against them.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Use

```bash
python -m operatorvoice.cli                       # synthetic corpus
python -m operatorvoice.cli my_feedback.csv       # your own CSV
python -m operatorvoice.cli my_feedback.csv --k 8 --seed 1 --out results_run1
```

Then open `out/report.html` in a browser. Options: `--k`, `--seed`, `--feasibility path.yaml`,
`--bootstrap N`, `--no-report`, `-v`.

Input CSV: a `text` column is required; `rating` (1-5, may be empty) and `source` are optional. Ratings
outside 1-5 are rejected. Use only feedback you are allowed to analyse and anonymise personal data first.

To adapt the feasibility rubric to your own product, copy `config/feasibility.yaml`, edit the term lists, and
pass it with `--feasibility`.

## Checks

```bash
ruff check . && ruff format --check .
pytest -q                          # 15 tests
python -m operatorvoice.evaluate 5 # compare against planted themes, writes results/
```

CI (`.github/workflows/ci.yml`) runs lint, format check, tests, a 3-seed evaluation, a report generation
run, and `pip-audit` on the runtime dependencies.

## Results on the synthetic corpus

Mean over 5 seeds, about 530 posts each, 10 planted themes (`python -m operatorvoice.evaluate 5`):

| Metric | Value |
|---|---|
| clusters chosen | 12.0 |
| adjusted Rand index vs planted themes | 0.554 |
| share of posts in their theme's main cluster | 0.718 |
| Spearman: predicted vs planted priority | 0.712 |
| overlap of top-5 opportunities with planted top-5 | 0.84 |
| mean feasibility, software-addressable themes | 0.768 |
| mean feasibility, hardware themes | 0.0 |

About a third of the synthetic posts mix in a remark about another theme, so moderate cluster agreement is
expected. The generator, scoring rubric and evaluation were written together, so these numbers show that the
pipeline recovers a known answer, not that it works on real feedback.

Runtime, single run on a Windows machine (AMD64, 16 threads), Python 3.13, concatenated copies of the
synthetic corpus (so not representative of real vocabulary): 572 posts in about 2.5 s (includes library warm-up),
8,000 posts in about 7 s, 300 bootstrap resamples under 0.3 s. Not a benchmark; no repeated trials.

## Security and privacy

- Input is untrusted: row limit, rating validation, no evaluation of content.
- The report escapes embedded data and renders post text as text, so a malicious post cannot run script (tested).
- YAML is loaded with `safe_load`.
- Outputs contain post text. `out/` is git-ignored; do not commit or share outputs of private feedback.
- The tool makes no network requests and needs no credentials, so there is no `.env`.

## Limitations

- **No real feedback is included.** The corpus is generated from word lists and is far cleaner than real posts.
- Clustering is lexical. The same need described in different words can split across clusters; sentence embeddings would help and are not used.
- Silhouette picked the top of the 6-12 range in every evaluated seed, so the chosen k is not strong evidence.
- Pain is a small hand-written word list plus rating, not a trained sentiment model. Feasibility is a keyword rubric, not an engineering estimate.
- The bootstrap covers post sampling only, not clustering instability.
- The report's JavaScript has no automated test in this repository (it was checked by hand in jsdom, not in real browsers). There is no screenshot yet.
- Power BI files are not included; the CSV and SQLite outputs are ready to import.

## Possible next steps

- Compare against a sentence-embedding clustering on a small, hand-labelled set of real public posts.
- Measure cluster stability across seeds and values of k and show it in the report.
- Add a browser test for the report in CI.

## Author

Abhinav Annareddy. MIT licence, see [LICENSE](LICENSE).
