"""Exercise parser + tools with NO LLM. Uses the CSV's own category as a stand-in until the Categorizer exists."""
import pandas as pd
from agents.parser import parse_csv, check_balance_continuity
import tools as T

tx, ex = parse_csv("data/sample_statement.csv")
print(f"kept={len(tx)} excluded={len(ex)} balance_breaks={check_balance_continuity(tx)}\n")
df = tx.rename(columns={"true_category": "category"})  # stand-in
pd.set_option("display.width", 160)
print("TOTAL SPENT:", T.total_spent(df))
print("\nBREAKDOWN\n", T.category_breakdown(df))
print("\nMONTHLY\n", T.monthly_trend(df))
print("\nCATEGORY MoM\n", T.category_mom(df).head(8))
print("\nTOP MERCHANTS\n", T.top_merchants(df))
print("\nANOMALIES\n", T.find_anomalies(df))
print("\nOVERDRAFTS\n", T.overdraft_events(df))
print("\nCAB LOOKUP\n", T.find_transactions(df, text="rapido").head(3))
print("\ncalc:", T.calculator("12429.68 / 4 * 1.1"))
