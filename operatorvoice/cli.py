"""python -m operatorvoice.cli [feedback.csv] [options]   (without a file, runs on the synthetic corpus)"""

import argparse
import logging
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from . import synth
from .pipeline import analyse, bootstrap_priority, load_csv
from .report import write_report

log = logging.getLogger("operatorvoice")
POST_COLUMNS = ["text", "source", "rating", "cluster", "pain"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="operatorvoice", description=__doc__)
    p.add_argument("csv", nargs="?", help="feedback CSV with a 'text' column (default: synthetic corpus)")
    p.add_argument("--out", default="out", help="output directory (default: out)")
    p.add_argument("--k", type=int, help="fix the number of clusters instead of choosing by silhouette")
    p.add_argument("--seed", type=int, default=0, help="random seed for clustering and bootstrap")
    p.add_argument("--feasibility", help="path to a feasibility YAML (default: config/feasibility.yaml)")
    p.add_argument("--bootstrap", type=int, default=300, help="bootstrap resamples for the report (default: 300)")
    p.add_argument("--no-report", action="store_true", help="skip the HTML report")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        df = load_csv(args.csv) if args.csv else synth.generate(1)
        log.info("analysing %d posts", len(df))
        res = analyse(df, k=args.k, feasibility_path=args.feasibility, seed=args.seed)
    except (ValueError, OSError, KeyError) as e:  # input problems: short message, no traceback
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    opp, posts = res["opportunities"], res["posts"][POST_COLUMNS]
    opp.to_csv(out / "opportunities.csv", index=False)
    posts.to_csv(out / "posts_labelled.csv", index=False)
    with closing(sqlite3.connect(out / "operatorvoice.db")) as con:
        opp.to_sql("opportunities", con, if_exists="replace", index=False)
        posts.to_sql("posts", con, if_exists="replace", index=False)
    written = ["opportunities.csv", "posts_labelled.csv", "operatorvoice.db"]
    if not args.no_report:
        write_report(res, bootstrap_priority(res, args.bootstrap, args.seed), out / "report.html")
        written.append("report.html")
    print(opp[["rank", "top_terms", "posts", "pain", "feasibility", "priority"]].to_string(index=False))
    print(f"\nwritten to {out}/: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
