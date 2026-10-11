"""
diagnostic_all_seeds.py
=========================
Systematic version of the seed-123/seed-456 checks: for ALL 5 seeds,
across LODO-3 (train=CIC17+CIC18, test=UNSW) and both Table VII
conditions (single-source=200k CIC17, multi-source=100k CIC17+100k
CIC18), retrains DAE and records raw (pre-normalization) mean
reconstruction error separately for UNSW benign vs. attack flows.

Answers: is the benign/attack reversal found on seed 123 (LODO-3) and
seed 456 (Table VII single-source) a 2-seed coincidence, or does it
show up broadly (e.g. 4/5, 5/5 seeds)? That determines how strongly
you can state the claim in the Discussion section.

Resumable: writes each (config, seed) result to disk immediately, and
skips anything already done on restart — same pattern as
paper3_consolidated.py. If it crashes, just rerun it.

Output: results_fixed/diagnostic_reversal_check.csv
  columns: config, seed, stop_epoch, AUC, benign_mean, attack_mean,
           benign_median, attack_median, reversal (bool)

Expected time: 15 DAE training runs total (5 seeds x 3 configs),
~1-1.5 hrs each based on your LODO/Table-VII timings = roughly
15-22 hours. Resumable, so safe to run unattended / overnight,
and safe to Ctrl+C and resume later if needed.
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
OUT_DIR  = r"C:/MLProject/paper3_ensemble/results_fixed"
OUT_PATH = os.path.join(OUT_DIR, "diagnostic_reversal_check.csv")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS     = [42, 123, 456, 789, 999]
N_PER_ENV = 100_000
N_SINGLE  = 300_000


class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(18,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(8,16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),
            nn.Linear(32,64),nn.ReLU(),nn.Linear(64,18))
    def forward(self,x): return self.dec(self.enc(x))


def train_ae(model_class, Xtr, Xva, seed, noise=0.1, epochs=100, patience=10):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_class()
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    crit = nn.HuberLoss(delta=1.0)
    Xt = torch.FloatTensor(Xtr)
    Xv = torch.FloatTensor(Xva)
    best, pat, best_state = float('inf'), 0, None
    stop_epoch = epochs
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = Xt[idx[i:i+512]]
            n = torch.clamp(b + torch.randn_like(b)*noise, -10, 10)
            opt.zero_grad(); crit(model(n), b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), Xv).item()
        sch.step(vl)
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                stop_epoch = ep + 1
                break
    model.load_state_dict(best_state)
    return model, stop_epoch


def get_ae_scores(model, X):
    model.eval(); out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            out.append(((b - model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def append_row(row):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(OUT_PATH)
    df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)


def already_done(config, seed):
    if not os.path.exists(OUT_PATH):
        return False
    df = pd.read_csv(OUT_PATH)
    return ((df["config"] == config) & (df["seed"] == seed)).any()


def evaluate_and_log(config, seed, model, Xt_u, yt_u, stop_epoch):
    raw_scores = get_ae_scores(model, Xt_u)
    auc = roc_auc_score(yt_u, raw_scores)
    benign = raw_scores[yt_u == 0]
    attack = raw_scores[yt_u == 1]
    reversal = bool(attack.mean() < benign.mean())

    row = {
        "config": config, "seed": seed, "stop_epoch": stop_epoch,
        "AUC": round(float(auc), 4),
        "benign_mean": round(float(benign.mean()), 6),
        "attack_mean": round(float(attack.mean()), 6),
        "benign_median": round(float(np.median(benign)), 6),
        "attack_median": round(float(np.median(attack)), 6),
        "reversal": reversal,
    }
    append_row(row)
    log.info("  [%s] seed %d: AUC=%.4f benign_mean=%.4f attack_mean=%.4f reversal=%s — saved",
              config, seed, auc, benign.mean(), attack.mean(), reversal)


def main():
    log.info("Loading data...")
    d = np.load(os.path.join(DEDUP, "train_dedup.npz"))
    X_all = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n = len(X_all) // 3
    X_c17, X_c18, X_usw = X_all[:n], X_all[n:2*n], X_all[2*n:]

    d = np.load(os.path.join(DEDUP, "val_dedup.npz"))
    V_all = np.clip(d["X_val"], -10, 10).astype(np.float32)
    nv = len(V_all) // 3
    V_c17, V_c18, V_usw = V_all[:nv], V_all[nv:2*nv], V_all[2*nv:]

    d = np.load(os.path.join(DEDUP, "test_unsw_dedup.npz"))
    X_test_unsw = np.clip(d["X"], -10, 10).astype(np.float32)
    y_test_unsw = d["y"].astype(int)
    log.info("UNSW test: %d rows (%d benign, %d attack)",
              len(y_test_unsw), (y_test_unsw==0).sum(), (y_test_unsw==1).sum())

    # Same subsampling scheme as paper3_consolidated.py's load_all()
    rng = np.random.default_rng(0)
    idx17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    idxus = rng.choice(len(X_usw), N_PER_ENV, replace=False)
    vi17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)
    vius  = rng.choice(len(V_usw), N_PER_ENV//4, replace=False)

    idx_s  = rng.choice(len(X_c17), N_SINGLE, replace=False)
    vi_s   = rng.choice(len(V_c17), N_SINGLE//4, replace=False)

    idx_cd_s   = rng.choice(len(X_c17), N_PER_ENV*2, replace=False)
    idx_cd_m17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx_cd_m18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    vi_cd_s    = rng.choice(len(V_c17), N_PER_ENV//2, replace=False)
    vi_cd_m17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi_cd_m18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)

    # LODO-3 training data (CIC17+CIC18, held out UNSW) — uses idx17/idx18/vi17/vi18
    X_lodo3 = np.concatenate([X_c17[idx17], X_c18[idx18]])
    V_lodo3 = np.concatenate([V_c17[vi17],  V_c18[vi18]])

    # Table VII single-source (200k CIC17)
    X_cd_s = X_c17[idx_cd_s]
    V_cd_s = V_c17[vi_cd_s]

    # Table VII multi-source (100k CIC17 + 100k CIC18)
    X_cd_m = np.concatenate([X_c17[idx_cd_m17], X_c18[idx_cd_m18]])
    V_cd_m = np.concatenate([V_c17[vi_cd_m17],  V_c18[vi_cd_m18]])

    configs = [
        ("LODO-3",          X_lodo3, V_lodo3),
        ("Table7-single",   X_cd_s,  V_cd_s),
        ("Table7-multi",    X_cd_m,  V_cd_m),
    ]

    for config_name, Xtr, Xva in configs:
        log.info("="*60)
        log.info("Config: %s (train=%d, val=%d)", config_name, len(Xtr), len(Xva))
        for seed in SEEDS:
            if already_done(config_name, seed):
                log.info("  [%s] seed %d — already done, skipping", config_name, seed)
                continue
            log.info("  [%s] seed %d — training...", config_name, seed)
            model, stop_epoch = train_ae(DAE, Xtr, Xva, seed)
            log.info("    stopped at epoch %d", stop_epoch)
            evaluate_and_log(config_name, seed, model, X_test_unsw, y_test_unsw, stop_epoch)

    # Summary
    df = pd.read_csv(OUT_PATH)
    log.info("="*60)
    log.info("SUMMARY — reversal rate by config:")
    for config_name, _, _ in configs:
        sub = df[df["config"] == config_name]
        n_rev = sub["reversal"].sum()
        log.info("  %-16s %d/%d seeds show reversal (benign_mean > attack_mean)",
                  config_name, n_rev, len(sub))
    log.info("Full results in %s", OUT_PATH)


if __name__ == "__main__":
    main()
