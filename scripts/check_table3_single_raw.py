"""
check_table3_single_raw.py
=============================
Checks whether Table III's SINGLE condition (300k CIC17 only, tested
on UNSW) shows the same clipping-saturation problem found in LODO-3
and Table VII: does raw (unclipped) AUC reveal real signal that the
normalized/clipped AUC is masking toward 0.500-ish values?

This is structurally the same kind of severe-shift scenario (train
on CIC17 only, evaluate on a very different domain, UNSW) as Table
VII's single-source condition — but Table III's Single condition
uses 300k CIC17 flows (vs Table VII's 200k), so it's worth checking
separately rather than assuming the same numbers apply.

Resumable, same pattern as the other scripts.

Output: results_fixed/table3_single_raw_check.csv

Expected time: ~5 DAE+TrAE training runs, ~1-1.5 hrs each based on
Table III timings ≈ 5-7 hours total (300k flows is closer to LODO-3's
200k than Table VII's 200k, but larger — expect on the slower end,
similar to the original Table III Single-condition timings you saw:
seed 42 took ~2hrs for DAE ep100 + TrAE ep40).
"""

import os, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP    = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
OUT_DIR  = r"C:/MLProject/paper3_ensemble/results_fixed"
OUT_PATH = os.path.join(OUT_DIR, "table3_single_raw_check.csv")
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


class TransformerAE(nn.Module):
    def __init__(self, n_feat=18, d_model=64, n_heads=4, d_ff=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb    = nn.Embedding(n_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, 1))
        self.n_feat = n_feat
    def forward(self, x):
        tokens  = self.input_proj(x.unsqueeze(-1))
        pos     = torch.arange(self.n_feat, device=x.device)
        tokens  = tokens + self.pos_emb(pos).unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.decoder(encoded).squeeze(-1)


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


def sample_rows(X, n, seed):
    if len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx]


def norm(s, p1, p99):
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def append_row(row):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(OUT_PATH)
    df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)


def already_done(seed):
    if not os.path.exists(OUT_PATH):
        return False
    df = pd.read_csv(OUT_PATH)
    return (df["seed"] == seed).any()


def main():
    log.info("Loading data...")
    d = np.load(os.path.join(DEDUP, "train_dedup.npz"))
    X_all = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n = len(X_all) // 3
    X_c17 = X_all[:n]

    d = np.load(os.path.join(DEDUP, "val_dedup.npz"))
    V_all = np.clip(d["X_val"], -10, 10).astype(np.float32)
    nv = len(V_all) // 3
    V_c17 = V_all[:nv]

    d = np.load(os.path.join(DEDUP, "test_unsw_dedup.npz"))
    X_test_unsw = np.clip(d["X"], -10, 10).astype(np.float32)
    y_test_unsw = d["y"].astype(int)
    log.info("UNSW test: %d rows (%d benign, %d attack)",
              len(y_test_unsw), (y_test_unsw==0).sum(), (y_test_unsw==1).sum())

    # Replay the exact rng sequence from load_all() up through idx_s/vi_s
    # (the Single-condition subsample), so this matches what Table III
    # actually trained on. CIC17/CIC18/UNSW and their val splits are all
    # equal thirds of the same source arrays, so their lengths match `n`
    # and `nv` respectively — no need to reload data to get them.
    rng = np.random.default_rng(0)
    _idx17 = rng.choice(n, N_PER_ENV, replace=False)
    _idx18 = rng.choice(n, N_PER_ENV, replace=False)
    _idxus = rng.choice(n, N_PER_ENV, replace=False)
    _vi17  = rng.choice(nv, N_PER_ENV//4, replace=False)
    _vi18  = rng.choice(nv, N_PER_ENV//4, replace=False)
    _vius  = rng.choice(nv, N_PER_ENV//4, replace=False)

    idx_s = rng.choice(n, N_SINGLE, replace=False)      # <-- the one we need
    vi_s  = rng.choice(nv, N_SINGLE//4, replace=False)  # <-- and this one

    X_single = X_c17[idx_s]
    V_single = V_c17[vi_s]
    log.info("Single condition: %d train, %d val (300k CIC17)", len(X_single), len(V_single))

    for seed in SEEDS:
        if already_done(seed):
            log.info("seed %d — already done, skipping", seed)
            continue
        log.info("seed %d — training DAE + TrAE...", seed)
        dae, ep_d = train_ae(DAE, X_single, V_single, seed)
        trae, ep_t = train_ae(TransformerAE, X_single, V_single, seed)
        log.info("  stopped: DAE ep%d  TrAE ep%d", ep_d, ep_t)

        raw_d = get_ae_scores(dae, X_test_unsw)
        raw_t = get_ae_scores(trae, X_test_unsw)
        raw_auc_dae  = roc_auc_score(y_test_unsw, raw_d)
        raw_auc_trae = roc_auc_score(y_test_unsw, raw_t)

        trs_d = get_ae_scores(dae, sample_rows(X_single, 50000, seed))
        p1d, p99d = np.percentile(trs_d,1), np.percentile(trs_d,99)
        trs_t = get_ae_scores(trae, sample_rows(X_single, 50000, seed))
        p1t, p99t = np.percentile(trs_t,1), np.percentile(trs_t,99)
        vs_d = norm(get_ae_scores(dae, V_single), p1d, p99d)
        vs_t = norm(get_ae_scores(trae, V_single), p1t, p99t)
        tau_d, tau_t = np.percentile(vs_d,95), np.percentile(vs_t,95)

        sd = norm(raw_d, p1d, p99d)
        st = norm(raw_t, p1t, p99t)
        norm_auc_dae  = roc_auc_score(y_test_unsw, sd)
        norm_auc_trae = roc_auc_score(y_test_unsw, st)
        f1_dae  = f1_score(y_test_unsw, (sd>tau_d).astype(int), zero_division=0)
        f1_trae = f1_score(y_test_unsw, (st>tau_t).astype(int), zero_division=0)

        benign_d = raw_d[y_test_unsw == 0]; attack_d = raw_d[y_test_unsw == 1]

        row = {
            "seed": seed,
            "stop_epoch_DAE": ep_d, "stop_epoch_TrAE": ep_t,
            "raw_AUC_DAE": round(float(raw_auc_dae), 4),
            "raw_AUC_TrAE": round(float(raw_auc_trae), 4),
            "norm_AUC_DAE": round(float(norm_auc_dae), 4),
            "norm_AUC_TrAE": round(float(norm_auc_trae), 4),
            "F1_DAE": round(float(f1_dae), 4),
            "F1_TrAE": round(float(f1_trae), 4),
            "DAE_benign_mean": round(float(benign_d.mean()), 6),
            "DAE_attack_mean": round(float(attack_d.mean()), 6),
            "DAE_reversal": bool(attack_d.mean() < benign_d.mean()),
        }
        append_row(row)
        log.info("  raw_AUC: DAE=%.4f TrAE=%.4f | norm_AUC: DAE=%.4f TrAE=%.4f | reversal=%s — saved",
                  raw_auc_dae, raw_auc_trae, norm_auc_dae, norm_auc_trae, row["DAE_reversal"])

    log.info("All done. Results in %s", OUT_PATH)


if __name__ == "__main__":
    main()
