"""
targeted_raw_auc_fix.py
=========================
Targeted rerun covering ONLY LODO-3 and both Table VII conditions
(single-source, multi-source) — the configs where UNSW is the fully
excluded target and percentile-clip normalization was saturating
AUC toward 0.500 regardless of real underlying separation.

LODO-1 and LODO-2 are NOT touched — their normalized AUCs already
show real variation (0.64-0.98), so clipping isn't saturating there
and they don't need to be rerun. Table III isn't touched either, for
the same reason.

For each (config, seed), this:
  1. Retrains DAE and TrAE with the already-fixed training logic
     (seeded init, clean validation) and the already-fixed percentile
     sampling (random sample_rows, not [:50000]).
  2. Computes RAW AUC (from unclipped reconstruction error) for DAE
     and TrAE individually — this is the mathematically correct fix:
     AUC only depends on score rank order, and raw scores have no
     clipping-induced ties, so this cannot saturate to 0.500 the way
     the normalized version can.
  3. Also computes the existing normalized/clipped AUC + F1 (DAE,
     TrAE, Ens2, and Ens3/SVDD using existing SVDD checkpoints — no
     SVDD retraining needed) for comparison and for threshold-based
     reporting, since F1 and ensemble averaging genuinely need a
     common 0-1 scale across models with different score magnitudes.
     Raw AUC has no equivalent for Ens2/Ens3, since averaging two
     models' RAW scores (different scales/units) isn't meaningful
     without normalization — only individual-model raw AUC is valid.

Resumable: writes each (config, seed) result to disk immediately,
skips anything already done on restart.

Output: results_fixed/table_lodo3_table7_raw_fixed.csv

Expected time: ~9-10 hours (15 DAE+TrAE training runs total:
5 seeds x 3 configs), based on your LODO-3 (~9hr) + Table VII
(~35min) timings from the full run.
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
CKPT_DIR = r"C:/MLProject/paper3_ensemble/checkpoints"
OUT_DIR  = r"C:/MLProject/paper3_ensemble/results_fixed"
OUT_PATH = os.path.join(OUT_DIR, "table_lodo3_table7_raw_fixed.csv")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS     = [42, 123, 456, 789, 999]
N_PER_ENV = 100_000


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


class SVDDEncoder(nn.Module):
    def __init__(self, input_dim=18, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,64), nn.ReLU(),
            nn.Linear(64,32),        nn.ReLU(),
            nn.Linear(32,latent_dim, bias=False))
    def forward(self, x): return self.net(x)


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


def get_svdd_scores(encoder, c, X):
    encoder.eval(); out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            z = encoder(b)
            out.append(((z - c)**2).sum(dim=1).numpy())
    return np.concatenate(out)


def load_svdd(seed, cond, exp=None):
    name = f"svdd_seed{seed}_{cond}"
    if exp:
        name += f"_{exp}"
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        return None, None
    ck = torch.load(path, map_location='cpu')
    enc = SVDDEncoder()
    enc.load_state_dict(ck['encoder_state'])
    enc.eval()
    return enc, ck['center']


def sample_rows(X, n, seed):
    """FIXED: random seeded sample, not X[:n] (which was biased toward
    whichever domain came first in a concatenated array)."""
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


def already_done(config, seed):
    if not os.path.exists(OUT_PATH):
        return False
    df = pd.read_csv(OUT_PATH)
    return ((df["config"] == config) & (df["seed"] == seed)).any()


def run_one(config_name, Xtr, Xva, X_test, y_test, seed, svdd_load_args):
    log.info("  [%s] seed %d — training DAE + TrAE...", config_name, seed)
    dae, ep_d = train_ae(DAE, Xtr, Xva, seed)
    trae, ep_t = train_ae(TransformerAE, Xtr, Xva, seed)
    log.info("    stopped: DAE ep%d  TrAE ep%d", ep_d, ep_t)

    # ── RAW (unclipped) scores — the corrected primary metric ──
    raw_d_test = get_ae_scores(dae, X_test)
    raw_t_test = get_ae_scores(trae, X_test)
    raw_auc_dae  = roc_auc_score(y_test, raw_d_test)
    raw_auc_trae = roc_auc_score(y_test, raw_t_test)

    # ── Normalized/clipped scores — kept for F1/threshold/ensemble ──
    trs_d = get_ae_scores(dae, sample_rows(Xtr, 50000, seed))
    p1d, p99d = np.percentile(trs_d,1), np.percentile(trs_d,99)
    trs_t = get_ae_scores(trae, sample_rows(Xtr, 50000, seed))
    p1t, p99t = np.percentile(trs_t,1), np.percentile(trs_t,99)
    vs_d = norm(get_ae_scores(dae, Xva), p1d, p99d)
    vs_t = norm(get_ae_scores(trae, Xva), p1t, p99t)
    tau_d, tau_t = np.percentile(vs_d,95), np.percentile(vs_t,95)

    sd = norm(raw_d_test, p1d, p99d)
    st = norm(raw_t_test, p1t, p99t)
    norm_auc_dae  = roc_auc_score(y_test, sd)
    norm_auc_trae = roc_auc_score(y_test, st)
    f1_dae  = f1_score(y_test, (sd>tau_d).astype(int), zero_division=0)
    f1_trae = f1_score(y_test, (st>tau_t).astype(int), zero_division=0)

    se2 = (sd+st)/2; tau_e2 = (tau_d+tau_t)/2
    norm_auc_ens2 = roc_auc_score(y_test, se2)
    f1_ens2 = f1_score(y_test, (se2>tau_e2).astype(int), zero_division=0)

    row = {
        "config": config_name, "seed": seed,
        "stop_epoch_DAE": ep_d, "stop_epoch_TrAE": ep_t,
        "raw_AUC_DAE": round(float(raw_auc_dae), 4),
        "raw_AUC_TrAE": round(float(raw_auc_trae), 4),
        "norm_AUC_DAE": round(float(norm_auc_dae), 4),
        "norm_AUC_TrAE": round(float(norm_auc_trae), 4),
        "norm_AUC_Ens2": round(float(norm_auc_ens2), 4),
        "F1_DAE": round(float(f1_dae), 4),
        "F1_TrAE": round(float(f1_trae), 4),
        "F1_Ens2": round(float(f1_ens2), 4),
    }

    # ── SVDD (existing checkpoint, no retraining) — raw + norm + Ens3 ──
    svdd_enc, svdd_c = load_svdd(*svdd_load_args)
    if svdd_enc is not None:
        raw_sv_test = get_svdd_scores(svdd_enc, svdd_c, X_test)
        raw_auc_sv = roc_auc_score(y_test, raw_sv_test)

        trs_sv = get_svdd_scores(svdd_enc, svdd_c, sample_rows(Xtr, 50000, seed))
        p1sv, p99sv = np.percentile(trs_sv,1), np.percentile(trs_sv,99)
        vs_sv = norm(get_svdd_scores(svdd_enc, svdd_c, Xva), p1sv, p99sv)
        tau_sv = np.percentile(vs_sv, 95)
        ssv = norm(raw_sv_test, p1sv, p99sv)
        norm_auc_sv = roc_auc_score(y_test, ssv)
        f1_sv = f1_score(y_test, (ssv>tau_sv).astype(int), zero_division=0)

        se3 = (sd+st+ssv)/3; tau_e3 = (tau_d+tau_t+tau_sv)/3
        norm_auc_ens3 = roc_auc_score(y_test, se3)
        f1_ens3 = f1_score(y_test, (se3>tau_e3).astype(int), zero_division=0)

        row.update({
            "raw_AUC_SVDD": round(float(raw_auc_sv), 4),
            "norm_AUC_SVDD": round(float(norm_auc_sv), 4),
            "norm_AUC_Ens3": round(float(norm_auc_ens3), 4),
            "F1_SVDD": round(float(f1_sv), 4),
            "F1_Ens3": round(float(f1_ens3), 4),
        })
    else:
        log.warning("    No SVDD checkpoint found for %s — Ens3/SVDD columns left blank", svdd_load_args)

    append_row(row)
    log.info("    raw_AUC: DAE=%.4f TrAE=%.4f | norm_AUC: DAE=%.4f TrAE=%.4f Ens2=%.4f — saved",
              raw_auc_dae, raw_auc_trae, norm_auc_dae, norm_auc_trae, norm_auc_ens2)


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

    # Same subsampling scheme as paper3_consolidated.py's load_all(),
    # replayed in full sequence so LODO-3/Table VII subsamples match
    # what the main pipeline used.
    rng = np.random.default_rng(0)
    idx17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    idxus = rng.choice(len(X_usw), N_PER_ENV, replace=False)
    vi17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)
    vius  = rng.choice(len(V_usw), N_PER_ENV//4, replace=False)

    N_SINGLE = 300_000
    idx_s = rng.choice(len(X_c17), N_SINGLE, replace=False)
    vi_s  = rng.choice(len(V_c17), N_SINGLE//4, replace=False)

    idx_cd_s   = rng.choice(len(X_c17), N_PER_ENV*2, replace=False)
    idx_cd_m17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx_cd_m18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    vi_cd_s    = rng.choice(len(V_c17), N_PER_ENV//2, replace=False)
    vi_cd_m17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi_cd_m18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)

    X_lodo3 = np.concatenate([X_c17[idx17], X_c18[idx18]])
    V_lodo3 = np.concatenate([V_c17[vi17],  V_c18[vi18]])

    X_cd_s = X_c17[idx_cd_s]
    V_cd_s = V_c17[vi_cd_s]

    X_cd_m = np.concatenate([X_c17[idx_cd_m17], X_c18[idx_cd_m18]])
    V_cd_m = np.concatenate([V_c17[vi_cd_m17],  V_c18[vi_cd_m18]])

    configs = [
        # (name, Xtr, Xva, svdd_load_args)
        ("LODO-3",        X_lodo3, V_lodo3, ("LODO-3", "UNSW")),   # matches load_svdd(seed, exp, held_out)
        ("Table7-single", X_cd_s,  V_cd_s,  None),   # no dedicated SVDD checkpoint for this ad-hoc condition
        ("Table7-multi",  X_cd_m,  V_cd_m,  None),
    ]

    for config_name, Xtr, Xva, svdd_args in configs:
        log.info("="*60)
        log.info("Config: %s (train=%d, val=%d)", config_name, len(Xtr), len(Xva))
        for seed in SEEDS:
            if already_done(config_name, seed):
                log.info("  [%s] seed %d — already done, skipping", config_name, seed)
                continue
            if svdd_args is not None:
                load_args = (seed,) + svdd_args
            else:
                load_args = (seed, "__no_checkpoint__")   # will just return (None, None)
            run_one(config_name, Xtr, Xva, X_test_unsw, y_test_unsw, seed, load_args)

    log.info("="*60)
    log.info("All done. Results in %s", OUT_PATH)


if __name__ == "__main__":
    main()
