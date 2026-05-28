#!/usr/bin/env python3
import sys
import pandas as pd

pf = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else 'logs/split_columns.txt'
df = pd.read_feather(pf)
with open(out, 'w') as f:
    for i, c in enumerate(df.columns):
        f.write(f"{i}\t{c}\n")
    f.write(f"TOTAL_COLS\t{len(df.columns)}\n")
print(f"Wrote columns to {out}")
