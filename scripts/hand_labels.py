"""Export a blind sample for hand labelling, and compare hand labels with the judge.

1) Export 150 explanations (condition hidden, order shuffled):
       python scripts/hand_labels.py export --run results/qwen2.5-3b-instruct
   -> hand_label_sample.csv with an empty `hand_label` column.
   Fill it with: detects / hedges / confabulates / incoherent (same rubric as judge_p2.py).

2) Agreement with the LLM judge:
       python scripts/hand_labels.py agree --run results/qwen2.5-3b-instruct
   Prints Cohen's kappa and a confusion table. Aim for kappa >= 0.7 before trusting the judge.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def export(run: Path, n: int, seed: int):
    df = pd.read_csv(run / "transcripts_judged.csv")
    use = df[(df.condition.isin(["m0_natural", "m1_swap"])) | df.is_flipped]
    per = max(1, n // use.condition.nunique())
    sample = (use.groupby("condition", group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), per), random_state=seed))
                 .sample(frac=1, random_state=seed))
    sample["row_id"] = sample.index
    blind = sample[["row_id", "question", "answer", "explanation"]].copy()
    blind["hand_label"] = ""
    blind.to_csv(run / "hand_label_sample.csv", index=False)
    print(f"wrote {len(blind)} rows to {run / 'hand_label_sample.csv'}")


def agree(run: Path):
    hand = pd.read_csv(run / "hand_label_sample.csv").dropna(subset=["hand_label"])
    judged = pd.read_csv(run / "transcripts_judged.csv")
    m = hand.merge(judged[["judge_label"]], left_on="row_id", right_index=True)
    m["hand_label"] = m.hand_label.str.strip().str.lower()
    print(f"n = {len(m)}, Cohen's kappa = {cohen_kappa_score(m.hand_label, m.judge_label):.3f}")
    print(pd.crosstab(m.hand_label, m.judge_label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "agree"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    export(Path(a.run), a.n, a.seed) if a.mode == "export" else agree(Path(a.run))


if __name__ == "__main__":
    main()
