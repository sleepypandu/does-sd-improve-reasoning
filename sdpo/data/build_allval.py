"""Build one validation parquet holding the test splits of all five SciKnowEval / ToolAlpaca domains.

verl groups validation metrics by data_source (val-core/<data_source>/...), and the four SciKnowEval
domains share data_source "sciknoweval", so each of their rows is relabelled sciknoweval_<domain>;
the feedback reward routes that prefix to the same MCQ scorer. ToolAlpaca keeps "tooluse". Prompts,
ground truths and extra_info are unchanged. SciKnowEval's unused embedding column is dropped so all
rows share one schema. Run after data/preprocess.py has written each domain's test.parquet.

  python data/build_allval.py      # -> datasets/allval/test.parquet
"""

import os

import pandas as pd

SDPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = [
    ("sciknoweval/biology", "sciknoweval_biology"),
    ("sciknoweval/chemistry", "sciknoweval_chemistry"),
    ("sciknoweval/material", "sciknoweval_material"),
    ("sciknoweval/physics", "sciknoweval_physics"),
    ("tooluse", "tooluse"),
]


def main():
    parts = []
    for rel, source in DOMAINS:
        df = pd.read_parquet(os.path.join(SDPO_DIR, "datasets", rel, "test.parquet"))
        assert df.data_source.nunique() == 1, rel
        parts.append(df.drop(columns=["embedding"], errors="ignore").assign(data_source=source))
        print(f"{rel:22} {len(df):4} rows -> data_source={source}")
    out = os.path.join(SDPO_DIR, "datasets", "allval", "test.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out)
    print(f"{out}: {sum(map(len, parts))} rows")


if __name__ == "__main__":
    main()
