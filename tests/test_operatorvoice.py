import numpy as np
import pandas as pd
import pytest

from operatorvoice import synth
from operatorvoice.cli import main
from operatorvoice.evaluate import evaluate_one
from operatorvoice.pipeline import analyse, bootstrap_priority, clean, load_csv, priority_score
from operatorvoice.report import write_report


def test_generator_is_reproducible_and_labelled():
    a, b = synth.generate(3), synth.generate(3)
    assert a.equals(b)
    assert set(a.theme) == set(synth.THEMES) | {"none"}
    assert len(a) > 500


def test_clean_removes_urls_and_punctuation():
    out = clean("Check https://x.io/a NOW!!! It's bad.")
    assert "http" not in out and "!" not in out and out.split() == ["check", "now", "it's", "bad"]


def test_analyse_returns_ranked_opportunities():
    res = analyse(synth.generate(1))
    opp = res["opportunities"]
    assert opp["rank"].tolist() == list(range(1, len(opp) + 1))
    assert opp["priority"].is_monotonic_decreasing
    assert opp.posts.sum() == len(res["posts"])
    assert {"top_terms", "pain", "feasibility", "example"} <= set(opp.columns)


def test_hardware_needs_are_not_ranked_as_software_opportunities():
    res = analyse(synth.generate(2))
    top3 = res["opportunities"].head(3)
    assert (top3.feasibility > 0.5).all()
    seats = res["opportunities"][res["opportunities"].top_terms.str.contains("seat|lumbar|noise|tyre", regex=True)]
    assert (seats.feasibility < 0.2).all()


def test_pipeline_recovers_planted_structure_better_than_chance():
    m = evaluate_one(1)
    assert m["ari"] > 0.35 and m["priority_spearman"] > 0.4 and m["top5_overlap"] >= 0.6
    assert m["sw_feasibility"] > m["hw_feasibility"] + 0.4


def test_load_csv_checks_columns(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"body": ["a"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_csv(p)
    pd.DataFrame({"text": ["the app crashes"], "rating": [1]}).to_csv(p, index=False)
    assert load_csv(p).iloc[0].rating == 1


def test_small_corpus_does_not_crash():
    df = synth.generate(1).head(40)
    res = analyse(df)
    assert len(res["posts"]) == 40


def test_rating_raises_pain():
    texts = [
        "the app is wrong and missing data",
        "the screen is wrong and fails",
        "map is missing and wrong again",
        "the app fails and is wrong",
        "great screen and the app is nice",
        "the map is clear and works",
        "app works well and looks nice",
        "good display and a helpful map",
    ]
    df = pd.DataFrame({"text": texts, "rating": [1] * 4 + [5] * 4})
    posts = analyse(df, k=2)["posts"]
    assert posts[posts.rating == 1].pain.mean() > posts[posts.rating == 5].pain.mean()


def test_load_csv_drops_empty_text_and_rejects_bad_ratings(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"text": ["the app crashes", None, "  "], "rating": [1, 2, 3]}).to_csv(p, index=False)
    assert load_csv(p).text.tolist() == ["the app crashes"]  # no literal "nan" post
    pd.DataFrame({"text": ["a b", "c d"], "rating": [7, 10]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="between 1 and 5"):
        load_csv(p)


def test_load_csv_row_limit(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"text": ["x y"] * 5}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="more than 3 rows"):
        load_csv(p, max_rows=3)


def test_too_few_posts_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        analyse(pd.DataFrame({"text": ["the app fails"] * 3, "rating": [1] * 3}))


def test_priority_score_weights():
    assert priority_score(0.5, 0.5, 0.5) == pytest.approx(0.125)
    assert priority_score(0.5, 0.5, 0.0) == 0.0  # zero feasibility removes the need
    assert priority_score(0.5, 0.4, 0.3, (0, 1, 1)) == pytest.approx(0.12)


def test_bootstrap_is_reproducible_and_brackets_estimate():
    res = analyse(synth.generate(1))
    a, b = bootstrap_priority(res, 100, seed=3), bootstrap_priority(res, 100, seed=3)
    assert a.equals(b)
    opp = res["opportunities"].merge(a, on=["rank", "cluster"])
    assert (opp.priority_lo <= opp.priority_hi).all()
    # allow a little slack: the point estimate is not the bootstrap median
    assert ((opp.priority >= opp.priority_lo - 0.05) & (opp.priority <= opp.priority_hi + 0.05)).all()
    assert a.p_top_n.between(0, 1).all()


def test_report_escapes_script_breakout(tmp_path):
    texts = ["the app fails </script><script>alert(1)</script> badly"] + synth.generate(1).text.head(60).tolist()
    res = analyse(pd.DataFrame({"text": texts, "rating": np.nan}), k=6)
    path = write_report(res, bootstrap_priority(res, 20), tmp_path / "r.html")
    html = path.read_text(encoding="utf-8")
    assert html.count("<script") == 2 and html.count("</script>") == 2  # data block + app script only


def test_cli_end_to_end_and_error_exit(tmp_path, capsys):
    assert main(["--out", str(tmp_path), "--bootstrap", "20"]) == 0
    assert {"opportunities.csv", "posts_labelled.csv", "operatorvoice.db", "report.html"} <= {
        f.name for f in tmp_path.iterdir()
    }
    assert main([str(tmp_path / "missing.csv"), "--out", str(tmp_path)]) == 2
    assert "error:" in capsys.readouterr().err
