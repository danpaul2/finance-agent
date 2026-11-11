"""Score the Categorizer against the CSV's original labels (ground truth).

  python evaluate.py              # rules + LLM  (needs API key)
  python evaluate.py --no-rules   # LLM only: the honest measure of the model
  python evaluate.py --rules-only # no LLM calls: coverage/accuracy of the rule pass alone
"""
import sys
import pandas as pd
from agents.parser import parse_csv
from agents.categorizer import categorize

use_rules = "--no-rules" not in sys.argv
rules_only = "--rules-only" in sys.argv

tx, _ = parse_csv("data/sample_statement.csv")
truth = tx["true_category"]
base = tx.drop(columns=["true_category"])

if rules_only:
    from agents.categorizer import rule_category
    base = base.copy()
    base["category"] = base.apply(rule_category, axis=1)
    base["cat_source"] = base["category"].map(lambda c: "rule" if c else None)
    hit = base["category"].notna()
    print(f"rules cover {hit.sum()}/{len(base)} rows ({hit.mean():.0%}); "
          f"accuracy on covered rows: {(base.loc[hit,'category']==truth[hit]).mean():.1%}")
    res = base[hit]; t = truth[hit]
else:
    res = categorize(base, use_rules=use_rules); t = truth
    print(f"overall accuracy: {(res['category']==t).mean():.1%}  (rules={'on' if use_rules else 'off'})")
    print(res["cat_source"].value_counts().to_string(), "\n")

mask = (res["category"] != t).to_numpy()
wrong = res[mask].assign(truth=t[mask].to_numpy())
print(f"{len(wrong)} mismatches")
if len(wrong):
    print(wrong.groupby(["description", "merchant", "truth", "category"]).size().rename("n").to_string())
