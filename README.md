# OperatorVoice - Feature Opportunities from Feedback

Analysis of public feedback from drivers, operators and car owners. It groups the unmet needs
people describe and ranks **software-feature opportunities** by demand, pain and feasibility.

## How it works

1. **Load** feedback (`text`, optional `rating` and `source`) from a CSV, or use the synthetic corpus.
2. **Group** the posts: TF-IDF (words and word pairs), truncated SVD, then k-means. The number of
   clusters is chosen by silhouette score between 6 and 12. Each cluster gets top terms as a label.
3. **Score** each cluster:
   - *demand*: posts in the cluster relative to the largest cluster
   - *pain*: share of negative wording, combined with the rating when there is one
   - *feasibility*: how many posts mention software-addressable things (app, update, notification,
     route, sync) minus hardware things (seat, tyre, noise, paint). The word lists are in
     `config/feasibility.yaml`, so you can change them.
4. **Rank** by `priority = demand x pain x feasibility`.
5. **Export** `out/opportunities.csv` and `out/posts_labelled.csv` (for Power BI or Excel) and a SQLite
   database with the same tables.

```bash
pip install -r requirements.txt
pytest -q
python -m operatorvoice.cli                 # synthetic corpus
python -m operatorvoice.cli my_feedback.csv # your own CSV
python -m operatorvoice.evaluate 5          # check against planted themes
```

## Results (latest local run, 5 seeds, about 530 synthetic posts each, 10 planted themes)

| Metric | Mean over seeds |
|---|---|
| clusters chosen | 12.0 |
| adjusted Rand index vs planted themes | 0.554 |
| share of posts in their theme's main cluster | 0.718 |
| Spearman: predicted vs planted priority | 0.712 |
| overlap of top-5 opportunities with planted top-5 | 0.84 |
| mean feasibility, software-addressable themes | 0.768 |
| mean feasibility, hardware themes | 0.0 |

The synthetic posts mix a main need with an unrelated remark in about a third of cases, so
moderate cluster agreement is expected. The ranking is more reliable than the exact cluster edges.
Re-run the command before quoting numbers.

## Limits (read before using this on a CV)

- **No real feedback is included.** The corpus is generated from word lists, so it is far cleaner
  than real posts. Results say the pipeline works on a known answer, not that it finds the
  right needs in real data. If you analyse real posts, check each site's terms of use and
  anonymise personal data first.
- Clustering is lexical (TF-IDF). Two posts that describe the same need in different words can land in
  different clusters. Sentence embeddings would help and are not used.
- Pain uses a small hand-written word list and the rating. It is not a trained sentiment model.
- Feasibility is a keyword rubric, not an engineering estimate.
- **Power BI is not included.** The CSV files are ready to import, but no report file is provided.

## Author

Abhinav Annareddy
