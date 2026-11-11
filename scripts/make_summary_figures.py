"""Cross-model summary table and figures from results/<model>/summary.csv.

    python scripts/make_summary_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "figures"
MODELS = {"qwen2.5-1.5b-instruct": "Qwen2.5-1.5B", "qwen2.5-3b-instruct": "Qwen2.5-3B",
          "smollm2-1.7b-instruct": "SmolLM2-1.7B"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load():
    rows = []
    for m, name in MODELS.items():
        s = pd.read_csv(ROOT / m / "summary.csv")
        get = lambda c, a=0.0: s[(s.condition == c) & (s.alpha == a)].iloc[0]
        m0, m1, m3 = get("m0_natural"), get("m1_swap"), get("m3_cue")
        rows.append({
            "model": name,
            "justify_own": m0.judge_confabulates,
            "justify_swapped": m1.judge_confabulates,
            "justify_swapped_ci_lo": m1.judge_confab_ci_lo, "justify_swapped_ci_hi": m1.judge_confab_ci_hi,
            "detects_swapped": m1.judge_detects,
            "disavow_keyword_swapped": m1.disavow_keyword_rate,
            "p_changed_own": m0.p_changed, "p_changed_swapped": m1.p_changed,
            "auroc_changed_swap_vs_own": m1.auroc_changed_vs_m0,
            "auroc_unusual_swap_vs_own": m1.auroc_unusual_vs_m0,
            "hint_followed_rate": m3.n_used / m3.n,
            "hint_mentioned_rate": m3.cue_mention_rate,
        })
    return pd.DataFrame(rows)


def fig_main(t):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(t))
    ax = axes[0]
    ax.bar(x - 0.2, t.justify_own, 0.38, color=SERIES[0], label="Own answer (M0)")
    err = np.vstack([t.justify_swapped - t.justify_swapped_ci_lo, t.justify_swapped_ci_hi - t.justify_swapped])
    ax.bar(x + 0.2, t.justify_swapped, 0.38, color=SERIES[1], label="Secretly swapped answer (M1)")
    ax.errorbar(x + 0.2, t.justify_swapped, yerr=err, fmt="none", ecolor=INK, lw=1, capsize=3)
    ax.set_xticks(x, t.model)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share: confident justification", fontsize=9, color=MUTED)
    ax.set_title("Models justify answers they never gave", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    style(ax)

    ax = axes[1]
    ax.bar(x - 0.2, t.auroc_changed_swap_vs_own, 0.38, color=SERIES[2], label='"Did you choose it, or was it changed?"')
    ax.bar(x + 0.2, t.auroc_unusual_swap_vs_own, 0.38, color=SERIES[3], label='"Anything unusual about your answer?"')
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(2.55, 0.51, "no signal", fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(x, t.model)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC, swapped vs own answers", fontsize=9, color=MUTED)
    ax.set_title("Asked directly, only a weak signal", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "choice_blindness.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_hint(t):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.hint_followed_rate, 0.38, color=SERIES[1], label="Answer switched to the hinted option")
    ax.bar(x + 0.2, t.hint_mentioned_rate, 0.38, color=SERIES[0], label="Of those switched: explanation mentions the hint")
    for i, v in enumerate(t.hint_mentioned_rate):
        ax.text(i + 0.2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, t.model)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share", fontsize=9, color=MUTED)
    ax.set_title('"A friend told me the answer is X": followed, but rarely admitted', fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "hidden_hint.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = load()
    t.round(3).to_csv(ROOT / "summary_all_models.csv", index=False)
    fig_main(t)
    fig_hint(t)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
