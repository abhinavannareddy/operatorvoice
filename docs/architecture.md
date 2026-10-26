# Architecture

OperatorVoice is a batch pipeline with no server and no network access. Everything runs locally and
writes files.

```
 feedback.csv ──► load_csv ─┐
                            ├─► analyse ──► opportunities, posts ─┬─► CSV + SQLite
 synth.generate ────────────┘      │                              └─► bootstrap_priority ─► report.html
                                   ▼
                        config/feasibility.yaml
```

| Module | Responsibility |
|---|---|
| `pipeline.py` | CSV loading and validation, text cleaning, clustering, scoring, `priority_score`, `bootstrap_priority` |
| `synth.py` | Seeded synthetic corpus with 10 planted themes and their ground-truth scores |
| `evaluate.py` | Compares pipeline output to the planted themes over several seeds |
| `report.py` | Turns a result into one self-contained HTML file |
| `cli.py` | Argument parsing, file output, exit codes |

## Algorithm

1. **Clean**: lowercase, strip URLs and punctuation.
2. **Vectorise**: TF-IDF over unigrams and bigrams, sublinear term frequency, English stop words.
   Corpora under 100 posts skip document-frequency cut-offs (`min_df=1`, `max_df=1.0`) because the
   cut-offs would remove nearly every term.
3. **Reduce**: truncated SVD to at most 25 components, then L2-normalise so k-means distance behaves
   like cosine distance.
4. **Cluster**: k-means. When `k` is not given, k from 6 to 12 is chosen by highest silhouette score.
5. **Score** each cluster:
   - `demand` = posts / posts in the largest cluster.
   - `pain` = per post `min(1, negative words / 2)`; with a rating, `0.7 * that + 0.3 * rating_term`
     where the rating term is 1 for ratings of 1-2, 0.5 for 3, 0 otherwise. Cluster pain is the mean.
   - `feasibility` = `clip(share of posts with a software term - 0.8 * share with a hardware term, 0, 1)`.
     Terms come from `config/feasibility.yaml`.
6. **Rank** by `demand^a * pain^b * feasibility^c`, with a = b = c = 1 by default.

The product form is deliberate. A cluster with zero feasibility scores zero however many people complain,
which is what stops seat comfort and road noise outranking app problems.

## Explainability

Every post keeps the word lists that produced its scores (`neg_terms`, `sw_terms`, `hw_terms`). The report
shows them next to the post text and aggregates them per cluster, so a feasibility value can be traced to
the words behind it. The weights in the report are exponents, so the effect of caring more about pain than
demand can be seen without re-running anything.

## Uncertainty

`bootstrap_priority` holds the clusters fixed and resamples posts inside each cluster (default 300 times),
recomputing pain and feasibility. It reports a 5-95% range of priority and the share of resamples in
which a cluster lands in the top three. Cluster sizes, and therefore demand, are held fixed.

It measures how much the ranking depends on which posts were collected. It does not measure instability
from choosing a different k or seed. The report states this.

## Outputs

`opportunities` (one row per cluster) and `posts` (one row per post) are written as CSV and as SQLite
tables of the same names.

| opportunities column | Meaning |
|---|---|
| rank, cluster | position by priority; internal cluster id |
| top_terms | five highest-weight terms in the cluster centroid |
| posts, share_pct | cluster size |
| pain, feasibility, demand, priority | scores described above |
| example | highest-pain post in the cluster |

| posts column | Meaning |
|---|---|
| text, source, rating | input values (`source` is `unknown` if absent) |
| cluster, pain | assigned cluster and post-level pain |

## Input handling and security boundaries

The CSV file is untrusted. The tool:

- rejects files over 200,000 rows and ratings outside 1-5 instead of clipping them;
- drops empty text rather than analysing the string "nan";
- never evaluates input; text only reaches regex cleaning and scikit-learn;
- escapes `<` in the JSON embedded in the report and renders post text with `textContent`, so a post
  containing `</script>` or HTML cannot execute in the report (covered by a test).

The feasibility YAML is read with `yaml.safe_load`. It is trusted configuration, but safe loading means a
malicious file cannot construct arbitrary Python objects.

Privacy: `report.html` and the CSV/SQLite outputs contain post text. Treat them with the same
restrictions as the source feedback. `out/` is git-ignored.

## Failure behaviour

Input problems (`ValueError`, unreadable file, missing column) print `error: ...` to stderr and exit with
code 2. Programming errors are not caught and show a traceback.

## Testing strategy

| Test area | What it checks |
|---|---|
| Generator | Reproducible per seed; all planted themes present |
| Loading | Missing column, empty text, bad ratings, row limit |
| Scoring | Ranking order, hardware needs score low feasibility, rating raises pain, exponent maths |
| Recovery | Pipeline vs planted themes (ARI, Spearman, top-5 overlap) stay above set floors |
| Bootstrap | Reproducible with a seed, interval ordering, probabilities in [0, 1] |
| Report | Script-breakout escape; CLI end-to-end and error exit code |

The report's JavaScript is not covered by the pytest suite. It was checked manually in jsdom (re-weighting
changes the order, reset restores it, drill-down renders 40 posts, no script errors); that check is not
part of the repository.

## Trade-offs

- **TF-IDF over embeddings.** Runs offline, deterministic, no model download, and the cluster terms are
  directly inspectable. The cost is that paraphrases land in different clusters.
- **Keyword rubrics over a trained model for pain and feasibility.** Transparent and editable, but crude.
- **Static HTML report over a web app.** No server to secure or deploy; the cost is that interactions are
  limited to re-weighting and drill-down on precomputed data.
- **k-means with silhouette.** Simple and fast. Silhouette tends to pick the upper end of the range on this
  corpus (12 in every evaluated seed), so the chosen k is a hint rather than a finding.
