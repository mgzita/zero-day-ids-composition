"""
lodo3_rank_fusion.py
=====================
Final LODO-3 run (train=CIC17+CIC18, held-out=UNSW), 5 seeds.

For each seed:
  1. Retrain DAE and TrAE (fixed training: seeded init, clean validation).
  2. Load the existing LODO-3 SVDD checkpoint (no SVDD retraining).
  3. Save per-flow RAW scores for all three models to disk, so no
     future analysis of LODO-3 needs retraining.
  4. Compute:
       - raw AUC for DAE, TrAE, SVDD (no clipping, cannot saturate)
       - rank-fused AUC for Ens-2 (DAE+TrAE) and Ens-3 (all three)
       - ensemble gain G_ens = Ens AUC - best individual raw AUC

Rank fusion: each model's raw scores are converted to percentile ranks
over the held-out test set (ties get the average rank), then averaged.
Ranks are scale-free, so DAE's much larger reconstruction errors can't
dominate the average and nothing is clipped, so nothing saturates.

Caveat to state in the paper: ranks are computed over the target test
set, so this needs a batch of target-domain flows and is not a
per-flow deployable score. It is an evaluation-time fix for the
score-scale problem, not a deployment method.

Resumable: skips any seed already in the output CSV.

Outputs (in OUT_DIR):
  lodo3_rank_fusion.csv
  lodo3_scores/seed{seed}.npz   (y, dae, trae, svdd raw scores)

Expected time: ~7-8 hours (5 x DAE+TrAE training).
"""

import os, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP     = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
CKPT_DIR  = r"C:/MLProject/paper3_ensemble/checkpoints"
OUT_DIR   = r"C:/MLProject/paper3_ensemble/results_fixed"
SCORE_DIR = os.path.join(OUT_DIR, "lodo3_scores")
OUT_PATH  = os.path.join(OUT_DIR, "lodo3_rank_fusion.csv")
os.makedirs(SCORE_DIR, exist_ok=True)

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
    """Must match train_svdd_paper3.py exactly."""
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


def ae_scores(model, X):
    model.eval(); out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            out.append(((b - model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def svdd_scores(enc, c, X):
    enc.eval(); out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            z = enc(Xt[i:i+10000])
            out.append(((z - c)**2).sum(dim=1).numpy())
    return np.concatenate(out)


def load_svdd(seed):
    path = os.path.join(CKPT_DIR, f"svdd_seed{seed}_LODO-3_UNSW.pt")
    if not os.path.exists(path):
        log.warning("Missing SVDD checkpoint: %s", path)
        return None, None
    ck = torch.load(path, map_location='cpu')
    enc = SVDDEncoder()
    enc.load_state_dict(ck['encoder_state'])
    enc.eval()
    return enc, ck['center']


def rank_norm(s):
    """Percentile rank in (0, 1], ties averaged. Scale-free, no clipping."""
    return rankdata(s, method="average") / len(s)


def already_done(seed):
    if not os.path.exists(OUT_PATH):
        return False
    return (pd.read_csv(OUT_PATH)["seed"] == seed).any()


def main():
    log.info("Loading data...")
    d = np.load(os.path.join(DEDUP, "train_dedup.npz"))
    X_all = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n = len(X_all) // 3
    X_c17, X_c18 = X_all[:n], X_all[n:2*n]

    d = np.load(os.path.join(DEDUP, "val_dedup.npz"))
    V_all = np.clip(d["X_val"], -10, 10).astype(np.float32)
    nv = len(V_all) // 3
    V_c17, V_c18 = V_all[:nv], V_all[nv:2*nv]

    d = np.load(os.path.join(DEDUP, "test_unsw_dedup.npz"))
    X_test = np.clip(d["X"], -10, 10).astype(np.float32)
    y_test = d["y"].astype(int)
    log.info("UNSW test: %d rows (%d benign, %d attack)",
             len(y_test), (y_test==0).sum(), (y_test==1).sum())

    # Same first six draws as load_all() in paper3_consolidated.py, in
    # the same order, so LODO-3 trains on the identical subsample.
    rng = np.random.default_rng(0)
    idx17 = rng.choice(n,  N_PER_ENV, replace=False)
    idx18 = rng.choice(n,  N_PER_ENV, replace=False)
    _idxus = rng.choice(n, N_PER_ENV, replace=False)
    vi17  = rng.choice(nv, N_PER_ENV//4, replace=False)
    vi18  = rng.choice(nv, N_PER_ENV//4, replace=False)

    Xtr = np.concatenate([X_c17[idx17], X_c18[idx18]])
    Xva = np.concatenate([V_c17[vi17],  V_c18[vi18]])
    log.info("LODO-3 train=%d val=%d", len(Xtr), len(Xva))

    for seed in SEEDS:
        if already_done(seed):
            log.info("seed %d — already done, skipping", seed)
            continue
        log.info("seed %d — training DAE + TrAE...", seed)
        dae, ep_d = train_ae(DAE, Xtr, Xva, seed)
        trae, ep_t = train_ae(TransformerAE, Xtr, Xva, seed)
        log.info("  stopped: DAE ep%d  TrAE ep%d", ep_d, ep_t)

        s_dae  = ae_scores(dae,  X_test)
        s_trae = ae_scores(trae, X_test)

        enc, c = load_svdd(seed)
        s_svdd = svdd_scores(enc, c, X_test) if enc is not None else None

        # Save per-flow scores so nothing here needs retraining again
        save = {"y": y_test, "dae": s_dae, "trae": s_trae}
        if s_svdd is not None:
            save["svdd"] = s_svdd
        np.savez_compressed(os.path.join(SCORE_DIR, f"seed{seed}.npz"), **save)

        auc_dae  = roc_auc_score(y_test, s_dae)
        auc_trae = roc_auc_score(y_test, s_trae)

        r_dae, r_trae = rank_norm(s_dae), rank_norm(s_trae)
        ens2 = roc_auc_score(y_test, (r_dae + r_trae) / 2)
        best2 = max(auc_dae, auc_trae)

        row = {
            "seed": seed, "stop_epoch_DAE": ep_d, "stop_epoch_TrAE": ep_t,
            "raw_AUC_DAE":  round(float(auc_dae), 4),
            "raw_AUC_TrAE": round(float(auc_trae), 4),
            "rank_AUC_Ens2": round(float(ens2), 4),
            "G_ens2": round(float(ens2 - best2), 4),
        }

        if s_svdd is not None:
            auc_svdd = roc_auc_score(y_test, s_svdd)
            r_svdd = rank_norm(s_svdd)
            ens3 = roc_auc_score(y_test, (r_dae + r_trae + r_svdd) / 3)
            best3 = max(auc_dae, auc_trae, auc_svdd)
            row.update({
                "raw_AUC_SVDD": round(float(auc_svdd), 4),
                "rank_AUC_Ens3": round(float(ens3), 4),
                "G_ens3": round(float(ens3 - best3), 4),
            })

        pd.DataFrame([row]).to_csv(
            OUT_PATH, mode='a', header=not os.path.exists(OUT_PATH), index=False)
        log.info("  raw: DAE=%.4f TrAE=%.4f%s | rank-fused: Ens2=%.4f%s — saved",
                 auc_dae, auc_trae,
                 f" SVDD={row['raw_AUC_SVDD']:.4f}" if s_svdd is not None else "",
                 ens2,
                 f" Ens3={row['rank_AUC_Ens3']:.4f}" if s_svdd is not None else "")

    df = pd.read_csv(OUT_PATH)
    log.info("=" * 60)
    log.info("SUMMARY (mean +/- pop std, n=%d seeds)", len(df))
    for col in ["raw_AUC_DAE", "raw_AUC_TrAE", "raw_AUC_SVDD",
                "rank_AUC_Ens2", "rank_AUC_Ens3", "G_ens2", "G_ens3"]:
        if col in df.columns:
            v = df[col].values
            log.info("  %-14s %.3f +/- %.3f", col, v.mean(), v.std(ddof=0))
    log.info("Saved -> %s", OUT_PATH)


if __name__ == "__main__":
    main()
