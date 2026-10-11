"""
svdd_raw_auc.py
================
Computes RAW (unclipped) AUC for every existing Deep SVDD checkpoint.
No training: loads checkpoints from CKPT_DIR and scores the test sets.

Covers:
  Table III  : Single and Multi checkpoints, each scored on CIC17, CIC18, UNSW
  Table IV   : LODO-1/2/3 checkpoints, each scored on its held-out target

Compare the raw_AUC column against the SVDD normalized AUCs already in
results_fixed/table3_single_multi.csv (SVDD_AUC) and table4_lodo.csv
(model == SVDD) to see whether clipping saturation masked any signal.

Output: results_fixed/svdd_raw_auc.csv
Expected time: a few minutes (35 checkpoints x small MLP scoring).
"""

import os, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP    = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
CKPT_DIR = r"C:/MLProject/paper3_ensemble/checkpoints"
OUT_DIR  = r"C:/MLProject/paper3_ensemble/results_fixed"
OUT_PATH = os.path.join(OUT_DIR, "svdd_raw_auc.csv")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 999]


class SVDDEncoder(nn.Module):
    """Must match train_svdd_paper3.py exactly."""
    def __init__(self, input_dim=18, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,64), nn.ReLU(),
            nn.Linear(64,32),        nn.ReLU(),
            nn.Linear(32,latent_dim, bias=False))
    def forward(self, x): return self.net(x)


def load_ckpt(name):
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        log.warning("Missing checkpoint: %s", path)
        return None, None
    ck = torch.load(path, map_location='cpu')
    enc = SVDDEncoder()
    enc.load_state_dict(ck['encoder_state'])
    enc.eval()
    return enc, ck['center']


def svdd_scores(enc, c, X):
    out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            z = enc(Xt[i:i+10000])
            out.append(((z - c)**2).sum(dim=1).numpy())
    return np.concatenate(out)


def main():
    tests = {}
    for name, fname in [("CIC17","test_cic17_dedup.npz"),
                        ("CIC18","test_cic18_dedup.npz"),
                        ("UNSW", "test_unsw_dedup.npz")]:
        d = np.load(os.path.join(DEDUP, fname))
        tests[name] = (np.clip(d["X"], -10, 10).astype(np.float32),
                       d["y"].astype(int))
        log.info("Loaded %s: %d rows", name, len(tests[name][0]))

    rows = []

    # Table III: Single / Multi, scored on all three test sets
    for cond in ["Single", "Multi"]:
        for seed in SEEDS:
            enc, c = load_ckpt(f"svdd_seed{seed}_{cond}")
            if enc is None:
                continue
            for ds, (X, y) in tests.items():
                auc = roc_auc_score(y, svdd_scores(enc, c, X))
                rows.append({"table": "III", "config": cond, "seed": seed,
                             "target": ds, "raw_AUC_SVDD": round(float(auc), 4)})
                log.info("  III  %-6s seed %-3d %-5s raw_AUC=%.4f", cond, seed, ds, auc)

    # Table IV: LODO configs, scored on held-out target
    for exp, held_out in [("LODO-1","CIC17"), ("LODO-2","CIC18"), ("LODO-3","UNSW")]:
        X, y = tests[held_out]
        for seed in SEEDS:
            enc, c = load_ckpt(f"svdd_seed{seed}_{exp}_{held_out}")
            if enc is None:
                continue
            auc = roc_auc_score(y, svdd_scores(enc, c, X))
            rows.append({"table": "IV", "config": exp, "seed": seed,
                         "target": held_out, "raw_AUC_SVDD": round(float(auc), 4)})
            log.info("  IV   %-6s seed %-3d %-5s raw_AUC=%.4f", exp, seed, held_out, auc)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    log.info("=" * 60)
    log.info("SUMMARY (mean +/- pop std across seeds):")
    for (t, cfg, tgt), sub in df.groupby(["table", "config", "target"]):
        v = sub["raw_AUC_SVDD"].values
        log.info("  Table %-3s %-7s -> %-5s  %.3f +/- %.3f  (n=%d)",
                  t, cfg, tgt, v.mean(), v.std(ddof=0), len(v))
    log.info("Saved -> %s", OUT_PATH)


if __name__ == "__main__":
    main()
