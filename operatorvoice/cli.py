"""python -m operatorvoice.cli [feedback.csv]   (without a file, runs on the synthetic corpus)"""
import sqlite3
import sys
from pathlib import Path

from . import synth
from .pipeline import analyse, load_csv


def main():
    df = load_csv(sys.argv[1]) if len(sys.argv) > 1 else synth.generate(1)
    res = analyse(df)
    out = Path("out")
    out.mkdir(exist_ok=True)
    opp = res["opportunities"]
    opp.to_csv(out / "opportunities.csv", index=False)
    res["posts"][["text", "source", "rating", "cluster", "pain"]].to_csv(out / "posts_labelled.csv", index=False)
    con = sqlite3.connect(out / "operatorvoice.db")
    opp.to_sql("opportunities", con, if_exists="replace", index=False)
    res["posts"][["text", "source", "rating", "cluster", "pain"]].to_sql("posts", con, if_exists="replace", index=False)
    con.close()
    print(opp[["rank", "top_terms", "posts", "pain", "feasibility", "priority"]].to_string(index=False))
    print("\nwritten: out/opportunities.csv, out/posts_labelled.csv, out/operatorvoice.db")


if __name__ == "__main__":
    main()
