import pandas as pd
import pytest

from operatorvoice import synth
from operatorvoice.evaluate import evaluate_one
from operatorvoice.pipeline import analyse, clean, load_csv


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
    texts = ["the app is wrong and missing data", "the screen is wrong and fails", "map is missing and wrong again",
             "the app fails and is wrong", "great screen and the app is nice", "the map is clear and works",
             "app works well and looks nice", "good display and a helpful map"]
    df = pd.DataFrame({"text": texts, "rating": [1] * 4 + [5] * 4})
    posts = analyse(df, k=2)["posts"]
    assert posts[posts.rating == 1].pain.mean() > posts[posts.rating == 5].pain.mean()
