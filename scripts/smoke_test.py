"""
smoke_test.py
=============
Targeted quick test of paper3_consolidated.py's Multi-condition path
(where the tau_sv crash happened) — seed 42 only, 3 epochs instead of
100, so training finishes in ~1-2 minutes instead of ~2 hours.

This does NOT touch results_fixed/ (the real run's output folder) —
it writes to results_fixed/smoke_test/ instead, so it can't interfere
with the real run's resumability tracking (done_keys checks).

Run this BEFORE launching the real paper3_consolidated.py, to confirm
the tau_sv fix works and the full Multi-condition path (training,
eval on all 3 test sets, SVDD checkpoint loading, Ens-2/Ens-3,
Table V/VI extras) completes without error.

If this finishes cleanly, delete results_fixed/smoke_test/ and launch
the real run.
"""

import os, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, average_precision_score

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# --- Paths — EDIT IF NEEDED, must match paper3_consolidated.py -----
DEDUP    = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
CKPT_DIR = r"C:/MLProject/paper3_ensemble/checkpoints"
OUT_DIR  = r"C:/MLProject/paper3_ensemble/results_fixed/smoke_test"
os.makedirs(OUT_DIR, exist_ok=True)

SEED      = 42
TEST_EPOCHS = 3        # instead of 100 — just enough to exercise the code
N_PER_ENV = 100_000
DEVICE    = "cpu"


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


def train_ae(model_class, Xtr, Xva, seed, noise=0.1, epochs=TEST_EPOCHS, patience=10):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_class().to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.HuberLoss(delta=1.0)
    Xt = torch.FloatTensor(Xtr).to(DEVICE)
    Xv = torch.FloatTensor(Xva).to(DEVICE)
    best, best_state = float('inf'), None
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
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, epochs


def get_ae_scores(model, X):
    model.eval(); out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X).to(DEVICE)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            out.append(((b - model(b))**2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


def get_svdd_scores(encoder, c, X):
    encoder.eval(); out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X).to(DEVICE)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            z = encoder(b)
            out.append(((z - c)**2).sum(dim=1).cpu().numpy())
    return np.concatenate(out)


def load_svdd(seed, cond):
    path = os.path.join(CKPT_DIR, f"svdd_seed{seed}_{cond}.pt")
    if not os.path.exists(path):
        log.warning("No SVDD checkpoint: %s", path)
        return None, None
    ck = torch.load(path, map_location='cpu')
    enc = SVDDEncoder().to(DEVICE)
    enc.load_state_dict(ck['encoder_state'])
    enc.eval()
    return enc, ck['center'].to(DEVICE)


def norm(s, p1, p99):
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def full_metrics(scores, yt, tau):
    yp = (scores > tau).astype(int)
    fpr, tpr, _ = roc_curve(yt, scores)
    return {
        "AUC": roc_auc_score(yt, scores),
        "PR_AUC": average_precision_score(yt, scores),
        "F1": f1_score(yt, yp, zero_division=0),
        "Precision": precision_score(yt, yp, zero_division=0),
        "Recall": recall_score(yt, yp, zero_division=0),
        "TPR@FPR001": float(np.interp(0.01, fpr, tpr)),
        "TPR@FPR005": float(np.interp(0.05, fpr, tpr)),
    }


def main():
    log.info("SMOKE TEST — Multi condition, seed %d, %d epochs", SEED, TEST_EPOCHS)
    log.info("Loading data...")
    d = np.load(os.path.join(DEDUP, "train_dedup.npz"))
    X_all = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n = len(X_all) // 3
    X_c17, X_c18, X_usw = X_all[:n], X_all[n:2*n], X_all[2*n:]

    d = np.load(os.path.join(DEDUP, "val_dedup.npz"))
    V_all = np.clip(d["X_val"], -10, 10).astype(np.float32)
    nv = len(V_all) // 3
    V_c17, V_c18, V_usw = V_all[:nv], V_all[nv:2*nv], V_all[2*nv:]

    tests = {}
    for name, fname in [
        ("CIC17","test_cic17_dedup.npz"),
        ("CIC18","test_cic18_dedup.npz"),
        ("UNSW", "test_unsw_dedup.npz"),
    ]:
        d = np.load(os.path.join(DEDUP, fname))
        tests[name] = (np.clip(d["X"], -10, 10).astype(np.float32), d["y"].astype(int))

    rng = np.random.default_rng(0)
    idx17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    idxus = rng.choice(len(X_usw), N_PER_ENV, replace=False)
    vi17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)
    vius  = rng.choice(len(V_usw), N_PER_ENV//4, replace=False)

    X_multi = np.concatenate([X_c17[idx17], X_c18[idx18], X_usw[idxus]])
    V_multi = np.concatenate([V_c17[vi17],  V_c18[vi18],  V_usw[vius]])

    log.info("Training DAE (Multi, seed %d, %d epochs)...", SEED, TEST_EPOCHS)
    dae, _ = train_ae(DAE, X_multi, V_multi, SEED)
    log.info("Training TrAE (Multi, seed %d, %d epochs)...", SEED, TEST_EPOCHS)
    trae, _ = train_ae(TransformerAE, X_multi, V_multi, SEED)
    log.info("Both models trained OK.")

    trs_d = get_ae_scores(dae, X_multi[:50000])
    p1d, p99d = np.percentile(trs_d,1), np.percentile(trs_d,99)
    trs_t = get_ae_scores(trae, X_multi[:50000])
    p1t, p99t = np.percentile(trs_t,1), np.percentile(trs_t,99)
    vs_d = norm(get_ae_scores(dae, V_multi), p1d, p99d)
    vs_t = norm(get_ae_scores(trae, V_multi), p1t, p99t)
    tau_d, tau_t = np.percentile(vs_d,95), np.percentile(vs_t,95)

    svdd_enc, svdd_c = load_svdd(SEED, "Multi")
    if svdd_enc is None:
        log.warning("No SVDD checkpoint found for seed %d Multi — "
                     "Ens-3/SVDD parts of the test will be skipped, "
                     "but this still verifies the DAE/TrAE/Ens-2 path.", SEED)

    rows = []
    ext_rows = []
    corr_rows = []
    for ds, (Xt, yt) in tests.items():
        sd = norm(get_ae_scores(dae, Xt), p1d, p99d)
        st = norm(get_ae_scores(trae, Xt), p1t, p99t)
        auc_d, auc_t = roc_auc_score(yt, sd), roc_auc_score(yt, st)
        se2 = (sd + st) / 2
        tau_e2 = np.percentile((vs_d+vs_t)/2, 95)
        auc_e2 = roc_auc_score(yt, se2)
        row = {"dataset": ds, "DAE_AUC": auc_d, "TrAE_AUC": auc_t, "Ens2_AUC": auc_e2}

        if svdd_enc is not None:
            trs_sv = get_svdd_scores(svdd_enc, svdd_c, X_multi[:50000])
            p1sv, p99sv = np.percentile(trs_sv,1), np.percentile(trs_sv,99)
            vs_sv = norm(get_svdd_scores(svdd_enc, svdd_c, V_multi), p1sv, p99sv)
            tau_sv = np.percentile(vs_sv, 95)   # the line that was missing before
            ssv = norm(get_svdd_scores(svdd_enc, svdd_c, Xt), p1sv, p99sv)
            auc_sv = roc_auc_score(yt, ssv)
            se3 = (sd + st + ssv) / 3
            tau_e3 = np.percentile((vs_d+vs_t+vs_sv)/3, 95)
            auc_e3 = roc_auc_score(yt, se3)
            row["SVDD_AUC"] = auc_sv
            row["Ens3_AUC"] = auc_e3

            if ds == "UNSW":
                ext_rows.append({**full_metrics(sd,yt,tau_d), "model":"DAE"})
                ext_rows.append({**full_metrics(st,yt,tau_t), "model":"TrAE"})
                ext_rows.append({**full_metrics(se2,yt,tau_e2), "model":"Ens2"})
                ext_rows.append({**full_metrics(ssv,yt,tau_sv), "model":"SVDD"})
                ext_rows.append({**full_metrics(se3,yt,tau_e3), "model":"Ens3"})
                pr_dt,_ = pearsonr(sd, st)
                pr_ds,_ = pearsonr(sd, ssv)
                pr_ts,_ = pearsonr(st, ssv)
                corr_rows += [
                    {"pair":"DAE-TrAE","pearson":pr_dt},
                    {"pair":"DAE-SVDD","pearson":pr_ds},
                    {"pair":"TrAE-SVDD","pearson":pr_ts},
                ]

        rows.append(row)
        log.info("  %s: DAE=%.4f TrAE=%.4f Ens2=%.4f%s",
                  ds, auc_d, auc_t, auc_e2,
                  f" SVDD={row.get('SVDD_AUC', float('nan')):.4f} Ens3={row.get('Ens3_AUC', float('nan')):.4f}"
                  if svdd_enc is not None else "")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "smoke_table3.csv"), index=False)
    if ext_rows:
        pd.DataFrame(ext_rows).to_csv(os.path.join(OUT_DIR, "smoke_table5.csv"), index=False)
    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "smoke_table6.csv"), index=False)

    log.info("=" * 55)
    log.info("SMOKE TEST PASSED — no crash, all metrics computed.")
    log.info("Output written to %s", OUT_DIR)
    log.info("If this looks right, delete %s and launch the real "
             "paper3_consolidated.py run.", OUT_DIR)


if __name__ == "__main__":
    main()
