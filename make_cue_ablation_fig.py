"""Prompt-cue leave-one-out ablation figures (seeded, watertight).

Each variant drops {pvim, <cue>} from the optimised prompt, trains the WINNER
LoRA (fixed seed=0 so only the prompt differs), and is scored VLM-only under the
20-seed protocol. Selection signal = native val (set05/06); test (set03 @ TTE90)
is confirmation only. Lower F1 when a cue is removed => that cue is more
important. The dashed line is the keep-all-6 model (nothing removed).

Style matches make_ablation_figs.py (crimson/slate, hidden spines, dotted grid).
Run:  python future_work/make_cue_ablation_fig.py
  ->  future_work/abl_cue_ranking_val.png
      future_work/abl_cue_ranking_test.png
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CRIMSON, BLUE, SLATE, GREEN = "#A6192E", "#1F77B4", "#8A8D8F", "#2E7D32"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15.5, "axes.labelsize": 14.5})

N = 20
TH = np.round(np.arange(0.05, 0.96, 0.025), 4)
CUES = ["orientation", "gaze", "movement", "posture", "context", "speed"]


def _f1(g, p):
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 2 * pr * rc / max(pr + rc, 1e-9)


def _split(g, s):
    r = np.random.default_rng(s)
    po = r.permutation(np.where(g == 1)[0]); ne = r.permutation(np.where(g == 0)[0])
    return np.concatenate([po[:len(po) // 2], ne[:len(ne) // 2]]), \
           np.concatenate([po[len(po) // 2:], ne[len(ne) // 2:]])


def vlm_f1(fn):
    """Mean/std VLM-only F1 at the F1-optimal threshold, 20-seed protocol."""
    g, s = [], []
    with open(fn) as f:
        for r in csv.DictReader(f):
            g.append(int(float(r["ground_truth"]))); s.append(float(r["qwen_score"]))
    g, s = np.array(g), np.array(s)
    rows = []
    for seed in range(N):
        v, te = _split(g, seed)
        best = max(((_f1(g[v], (s[v] >= t).astype(int)), t) for t in TH), key=lambda x: x[0])[1]
        rows.append(_f1(g[te], (s[te] >= best).astype(int)))
    return float(np.mean(rows)), float(np.std(rows))


val = {c: vlm_f1(ROOT / f"results/promptstudy_val_drop_{c}.csv") for c in CUES}
test = {c: vlm_f1(ROOT / f"results/promptstudy_test_drop_{c}_tte90.csv") for c in CUES}
val_ceil = vlm_f1(ROOT / "results/promptstudy_val_base_nopvim.csv")[0]
test_ceil = vlm_f1(ROOT / "results/lora_opt_nopvim_2-4s_eval90_set03.csv")[0]

# Same cue order in both figures (ranked on val, the selection set) so they compare.
ORDER = sorted(CUES, key=lambda c: val[c][0])
TRIMMABLE = ORDER[-1]


def panel(means, errs, ceil, title, out):
    y = np.arange(len(ORDER))
    cols = [CRIMSON if c == "speed" else (GREEN if c == TRIMMABLE else SLATE) for c in ORDER]
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)
    ax.barh(y, means, xerr=errs, color=cols, zorder=3,
            error_kw=dict(ecolor="#555", capsize=3, lw=1.1))
    pad = (max(errs) if errs is not None else 0) + 0.012
    for yi, v in zip(y, means):
        ax.text(v + pad, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=11.5, fontweight="bold")
    ax.axvline(ceil, ls="--", color=BLUE, lw=1.8, zorder=4)
    ax.text(ceil, len(ORDER) - 0.35, f"keep all 6 = {ceil:.3f}", color=BLUE, fontsize=11,
            ha="center", va="bottom", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(ORDER)
    ax.set_ylim(-0.6, len(ORDER) + 0.35)
    ax.set_title(title)
    ax.set_xlabel("VLM-only F1 with the cue removed   (lower = cue more important)")
    ax.set_ylabel("cue removed from prompt")
    ax.set_xlim(0, max(max(means), ceil) * 1.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, ls=":", alpha=0.5, zorder=0, axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CRIMSON),
               plt.Rectangle((0, 0), 1, 1, color=GREEN),
               plt.Rectangle((0, 0), 1, 1, color=SLATE)]
    ax.legend(handles, ["most critical (speed)", "most trimmable", "other cues"],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {HERE / out}")


panel(np.array([val[c][0] for c in ORDER]), np.array([val[c][1] for c in ORDER]), val_ceil,
      "Prompt-cue ablation on the validation set (set05/06)", "abl_cue_ranking_val.png")
panel(np.array([test[c][0] for c in ORDER]), np.array([test[c][1] for c in ORDER]), test_ceil,
      "Prompt-cue ablation on the test set (set03, TTE 90)", "abl_cue_ranking_test.png")
