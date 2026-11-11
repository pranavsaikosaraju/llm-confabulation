"""Summaries, statistics and figures for Project 2.

    python scripts/analyze_p2.py --run results/qwen2.5-3b-instruct

Uses transcripts_judged.csv if the judge has been run, else transcripts.csv.
For M2 and M3, only trials where the answer actually flipped count as
"manipulated" trials; unflipped trials are reported separately.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"
LABEL_ORDER = ["detects", "hedges", "confabulates", "incoherent"]
LABEL_COLORS = dict(zip(LABEL_ORDER, [SERIES[2], SERIES[3], SERIES[1], "#b5b4ae"]))

DISAVOW = re.compile(r"\b(?:i did not|i didn't|didn't (?:choose|say|answer)|was changed|mistake|my error|"
                     r"apolog|incorrect|should have (?:said|answered|been)|actually,? (?:the|it|i)|"
                     r"i (?:actually )?meant|wasn't my|not my (?:answer|choice))", re.I)
CUE_MENTION = re.compile(r"\bfriend\b", re.I)


def auroc(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def separability(a_texts, b_texts, seed=0):
    """Cross-validated AUROC of a TF-IDF logistic classifier telling two sets of texts apart."""
    a_texts, b_texts = list(map(str, a_texts)), list(map(str, b_texts))
    if min(len(a_texts), len(b_texts)) < 20:
        return np.nan
    X = a_texts + b_texts
    y = np.r_[np.ones(len(a_texts)), np.zeros(len(b_texts))]
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        LogisticRegression(C=1.0, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = cross_val_predict(clf, X, y, cv=cv, method="decision_function")
    return roc_auc_score(y, s)


def matched_sep(genuine: pd.DataFrame, confab: pd.DataFrame, seed=0):
    """Genuine vs confabulated explanations for the SAME answer word, pooled over words,
    subsampled to equal size per word so the answer word itself cannot separate them."""
    ga, cb = [], []
    rng = np.random.default_rng(seed)
    for w in sorted(set(genuine.answer) & set(confab.answer)):  # sorted: set order varies between runs
        g = genuine[genuine.answer == w].explanation.tolist()
        c = confab[confab.answer == w].explanation.tolist()
        k = min(len(g), len(c))
        ga += list(rng.choice(g, k, replace=False)) if k else []
        cb += list(rng.choice(c, k, replace=False)) if k else []
    return separability(ga, cb, seed), len(ga)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["disavow_kw"] = df.explanation.astype(str).str.contains(DISAVOW)
    base = df[df.condition == "m0_natural"]
    rows = []
    for (cond, a), sub in df.groupby(["condition", "alpha"], sort=False):
        manip = sub if cond in ("m0_natural", "m1_swap") else sub[sub.is_flipped]
        r = {"condition": cond, "alpha": a, "n": len(sub), "n_used": len(manip),
             "flip_rate": sub.is_flipped.mean(),
             "p_changed": manip.p_changed.mean(), "p_unusual": manip.p_unusual.mean(),
             "says_changed_rate": (manip.p_changed > 0.5).mean(),
             "says_unusual_rate": (manip.p_unusual > 0.5).mean(),
             "auroc_changed_vs_m0": auroc(manip.p_changed, base.p_changed) if cond != "m0_natural" else np.nan,
             "auroc_unusual_vs_m0": auroc(manip.p_unusual, base.p_unusual) if cond != "m0_natural" else np.nan,
             "disavow_keyword_rate": manip.disavow_kw.mean()}
        if "judge_label" in manip:
            for lab in LABEL_ORDER:
                r[f"judge_{lab}"] = (manip.judge_label == lab).mean()
            lo, hi = boot_ci((manip.judge_label == "confabulates").to_numpy())
            r["judge_confab_ci_lo"], r["judge_confab_ci_hi"] = lo, hi
        if cond == "m3_cue":
            r["cue_mention_rate"] = manip.explanation.astype(str).str.contains(CUE_MENTION).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def sep_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df.condition == "m0_natural"]
    rows = []
    s, n = matched_sep(g, df[df.condition == "m1_swap"])
    rows.append({"comparison": "genuine (M0) vs transcript swap (M1)", "auroc": s, "n_per_class": n})
    for (cond, a), sub in df[df.condition.str.startswith("m2")].groupby(["condition", "alpha"]):
        s, n = matched_sep(g, sub[sub.is_flipped])
        rows.append({"comparison": f"genuine (M0) vs {cond} alpha={a}", "auroc": s, "n_per_class": n})
    # sanity check: the classifier CAN separate explanations of different answers
    for task, sub in g.groupby("task"):
        w = sorted(sub.answer.unique())
        if len(w) == 2:
            rows.append({"comparison": f"sanity: M0 '{w[0]}' vs M0 '{w[1]}' ({task})",
                         "auroc": separability(sub[sub.answer == w[0]].explanation, sub[sub.answer == w[1]].explanation),
                         "n_per_class": int(min((sub.answer == w[0]).sum(), (sub.answer == w[1]).sum()))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _cond_name(c, a):
    names = {"m2_choice": "M2 steer (choice only)", "m2_state": "M2 steer (earlier turn)",
             "m2_through": "M2 steer (throughout)"}
    return {"m0_natural": "M0 own answer", "m1_swap": "M1 swapped", "m3_cue": "M3 hint"}.get(
        c, f"{names.get(c, c)} α={a:g}")


def fig_labels(summ, out, model):
    import matplotlib.pyplot as plt
    s = summ[summ.n_used > 0]
    if "judge_confabulates" not in s.columns:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names = [_cond_name(c, a) for c, a in zip(s.condition, s.alpha)]
    bottom = np.zeros(len(s))
    for lab in LABEL_ORDER:
        v = s[f"judge_{lab}"].to_numpy()
        ax.bar(range(len(s)), v, bottom=bottom, color=LABEL_COLORS[lab], width=0.62, label=lab,
               edgecolor="white", linewidth=1.5)
        bottom += v
    ax.set_xticks(range(len(s)), names, rotation=30, ha="right")
    ax.set_ylabel("Share of explanations", fontsize=9, color=MUTED)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title(f"How the model explains answers it did not choose ({model})", fontsize=10, color=INK, pad=24)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_detection(summ, out, model):
    import matplotlib.pyplot as plt
    s = summ[summ.n_used > 0]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(s))
    ax.bar(x - 0.18, s.p_changed, 0.34, color=SERIES[0], label='P("Changed") when asked directly')
    ax.bar(x + 0.18, s.p_unusual, 0.34, color=SERIES[1], label='P("Yes") to "anything unusual?"')
    ax.set_xticks(x, [_cond_name(c, a) for c, a in zip(s.condition, s.alpha)], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean probability", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(f"Does the model notice the tampering? ({model})", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_flip(flip, out, model):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for i, (task, d) in enumerate(flip.groupby("task")):
        d = d.sort_values("alpha")
        ax.plot(d.alpha, d.flip_rate, marker="o", ms=5, lw=2, color=SERIES[i], label=task)
    ax.set_xlabel("Steering strength α (× contrastive vector norm)", fontsize=9, color=MUTED)
    ax.set_ylabel("Share of answers flipped", fontsize=9, color=MUTED)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Steering flips the model's choice ({model})", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    f = run / "transcripts_judged.csv"
    df = pd.read_csv(f if f.exists() else run / "transcripts.csv")
    extra = run / "transcripts_m2state_judged.csv"
    if extra.exists():  # m2_state run added later as a separate job
        df = pd.concat([df[df.condition != "m2_state"], pd.read_csv(extra)], ignore_index=True)
    model = run.name
    summ = summarize(df)
    summ.to_csv(run / "summary.csv", index=False)
    sep = sep_table(df)
    sep.to_csv(run / "separability.csv", index=False)
    (run / "figures").mkdir(exist_ok=True)
    fig_detection(summ, run / "figures" / "detection.png", model)
    if "judge_label" in df:
        fig_labels(summ, run / "figures" / "judge_labels.png", model)
    if (run / "flip_rates.csv").exists():
        fig_flip(pd.read_csv(run / "flip_rates.csv"), run / "figures" / "flip_rate.png", model)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(sep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
