"""Apply the thesis 20-seed val/test protocol to the LoRA adapter's scores.

Mirrors validated_tables.py exactly (same THRESHOLDS, BETAS, 20 stratified
50/50 val/test splits; tune the threshold / (beta, threshold) on the val half,
report on the held-out test half, mean +/- std). Makes the fine-tuned VLM
directly comparable to the thesis tables.

Input: a CSV with columns ground_truth, pvim_prob, qwen_score (continuous P(yes)),
e.g. produced by eval_lora_qwen.py on set03.

Run:
    python future_work/lora_validated.py \
        --csv results/qwen25_32b_pvim_lora_regbal_tte90_set03.csv
"""
import argparse
import csv
import numpy as np

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 4)
BETAS = np.round(np.arange(0.0, 1.001, 0.05), 3)
N_SEEDS = 20


def metrics(gt, pred):
    tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum()); tn = int(((pred == 0) & (gt == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(acc=(tp + tn) / max(len(gt), 1), prec=prec, rec=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-9),
                f2=5 * prec * rec / max(4 * prec + rec, 1e-9))


def stratified_split(gt, seed):
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.where(gt == 1)[0]); neg = rng.permutation(np.where(gt == 0)[0])
    val = np.concatenate([pos[:len(pos) // 2], neg[:len(neg) // 2]])
    test = np.concatenate([pos[len(pos) // 2:], neg[len(neg) // 2:]])
    return val, test


def tune_thr(gt_v, sc_v, key):
    best = (-1.0, 0.5)
    for t in THRESHOLDS:
        m = metrics(gt_v, (sc_v >= t).astype(int))
        if m[key] > best[0]:
            best = (m[key], t)
    return best[1]


def tune_fused(gt_v, pv_v, vl_v, key):
    best = (-1.0, 0.9, 0.5)
    for b in BETAS:
        fused = b * pv_v + (1 - b) * vl_v
        for t in THRESHOLDS:
            m = metrics(gt_v, (fused >= t).astype(int))
            if m[key] > best[0]:
                best = (m[key], b, t)
    return best[1], best[2]


def run_protocol(gt, score, pvim=None, fuse=False):
    """Return per-metric mean/std under F1- and F2-optimal tuning."""
    out = {}
    for key in ("f1", "f2"):
        rows = []
        betas = []
        for seed in range(N_SEEDS):
            v, te = stratified_split(gt, seed)
            if fuse:
                b, thr = tune_fused(gt[v], pvim[v], score[v], key)
                pred = (b * pvim[te] + (1 - b) * score[te] >= thr).astype(int)
                betas.append(b)
            else:
                thr = tune_thr(gt[v], score[v], key)
                pred = (score[te] >= thr).astype(int)
            rows.append(metrics(gt[te], pred))
        agg = {m: (np.mean([r[m] for r in rows]), np.std([r[m] for r in rows]))
               for m in ("f1", "f2", "acc", "prec", "rec")}
        agg["beta"] = float(np.mean(betas)) if betas else None
        out[key] = agg
    return out


def fmt(agg, score_key):
    m = agg
    beta = f"  beta={m['beta']:.2f}" if m["beta"] is not None else ""
    return (f"{score_key.upper()}={m[score_key][0]:.3f}+/-{m[score_key][1]:.3f}  "
            f"Acc={m['acc'][0]:.3f}  Prec={m['prec'][0]:.3f}  Rec={m['rec'][0]:.3f}{beta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    gt, score, pvim = [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            gt.append(int(float(row["ground_truth"])))
            score.append(float(row["qwen_score"]))
            pvim.append(float(row["pvim_prob"]))
    gt = np.array(gt); score = np.array(score); pvim = np.array(pvim)
    print(f"Loaded {len(gt)} samples ({gt.sum()} cross, {len(gt)-gt.sum()} no-cross)")
    print(f"  qwen_score range [{score.min():.3f}, {score.max():.3f}] "
          f"(continuous = good; all 0/1 = binary, threshold tuning limited)\n")

    pvim_res = run_protocol(gt, pvim)
    vlm_res = run_protocol(gt, score)
    fus_res = run_protocol(gt, score, pvim=pvim, fuse=True)

    print("================ 20-seed val/test protocol (TTE 90) ================")
    print("\n-- F1-optimal --")
    print(f"  PVIM-only      : {fmt(pvim_res['f1'], 'f1')}")
    print(f"  VLM-only (LoRA): {fmt(vlm_res['f1'], 'f1')}")
    print(f"  Fusion (LoRA)  : {fmt(fus_res['f1'], 'f1')}")
    print("\n-- F2-optimal --")
    print(f"  PVIM-only      : {fmt(pvim_res['f2'], 'f2')}")
    print(f"  VLM-only (LoRA): {fmt(vlm_res['f2'], 'f2')}")
    print(f"  Fusion (LoRA)  : {fmt(fus_res['f2'], 'f2')}")

    print("\n-- thesis reference (Standard, TTE 90) --")
    print("  PVIM-only      : F1=0.570  F2=0.659")
    print("  Zero-shot VLM  : F1=0.519  F2=0.503")
    print("  Fusion Standard: F1=0.577  F2=0.670")


if __name__ == "__main__":
    main()
