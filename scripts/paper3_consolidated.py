"""
paper3_consolidated.py
=======================
ONE script, fixed, covering everything DAE/TrAE-related for Paper 3
camera-ready. Replaces: train_paper3.py, run_lodo_paper3.py (all
versions), run_lodo_789_999.py, extended_analysis.py's DAE/TrAE parts.

Two bugs fixed relative to the original scripts:

  BUG 1 — Non-deterministic early stopping.
    Original: validation loss was computed with FRESH random noise
    added every epoch, so "is this epoch the best so far" was partly
    a coin flip. This could freeze training on an early, undertrained
    checkpoint (seen as AUC collapsing to exactly 0.500 on some seeds).
    Fix: validation loss computed on CLEAN (unnoised) data. Training-time
    denoising noise is UNCHANGED — that's a correct, intentional part
    of the DAE objective (Vincent et al.), only validation was wrong.

  BUG 2 — Unseeded weight initialization.
    Original: `model = DAE()` was constructed BEFORE `torch.manual_seed
    (seed)` was called, so a given seed did not reproduce the same
    initial weights across different scripts/call sequences. This made
    "seed 456" in one script incomparable to "seed 456" in another.
    Fix: model is constructed INSIDE train_ae(), after seeding.

SVDD is NOT retrained here — train_svdd_paper3.py already seeds
correctly and validates on clean data, so its existing checkpoints
in CKPT_DIR are reused directly to rebuild Ens-3.

Covers:
  Table III  — Single- vs Multi-environment (CIC17/CIC18/UNSW)
  Table IV   — LODO-1, LODO-2, LODO-3 (AUC + F1)
  Table V    — Extended metrics (PR-AUC, Precision, Recall, TPR@FPR)
  Table VI   — Pairwise score correlation
  Table VII  — Clean diversity experiment
  Ens-2 / Ens-3 — rebuilt everywhere, Ens-3 using existing SVDD checkpoints

Expected time: DAE+TrAE dominate cost. Single+Multi (2 conditions) +
3 LODO configs + 2 clean-diversity conditions = 7 conditions x 5 seeds
x 2 models = 70 training runs. At ~1-2 hrs/run on CPU this is a
multi-day job — run overnight/over a weekend, or on GPU if available
(set DEVICE below).

Outputs (all in OUT_DIR):
  table3_single_multi.csv
  table4_lodo.csv
  table5_extended_metrics.csv
  table6_score_correlation.csv
  table7_clean_diversity.csv
  paired_stats.csv
"""

import os, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, wilcoxon
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, roc_curve,
)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# --- Paths — EDIT IF NEEDED -----------------------------------------
DEDUP    = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
OUT_DIR  = r"C:/MLProject/paper3_ensemble/results_fixed"
CKPT_DIR = r"C:/MLProject/paper3_ensemble/checkpoints"   # existing SVDD checkpoints
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS     = [42, 123, 456, 789, 999]
NOISE     = 0.1
N_PER_ENV = 100_000
N_SINGLE  = 300_000
DEVICE    = "cpu"   # set to "cuda" if available


# ── Models (unchanged architectures) ─────────────────────────────
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
    def __init__(self, n_feat=18, d_model=64,
                 n_heads=4, d_ff=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb    = nn.Embedding(n_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, 1))
        self.n_feat = n_feat

    def forward(self, x):
        tokens  = self.input_proj(x.unsqueeze(-1))
        pos     = torch.arange(self.n_feat, device=x.device)
        tokens  = tokens + self.pos_emb(pos).unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.decoder(encoded).squeeze(-1)


class SVDDEncoder(nn.Module):
    """Must match train_svdd_paper3.py exactly, for checkpoint loading."""
    def __init__(self, input_dim=18, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,64), nn.ReLU(),
            nn.Linear(64,32),        nn.ReLU(),
            nn.Linear(32,latent_dim, bias=False))
    def forward(self, x): return self.net(x)


# ── FIXED training function ──────────────────────────────────────
def train_ae(model_class, Xtr, Xva, seed, noise=NOISE,
             epochs=100, patience=10):
    """
    Trains an autoencoder-style model (DAE or TransformerAE).

    Fix 1: model is constructed AFTER seeding -> reproducible init.
    Fix 2: validation loss computed on CLEAN Xva -> deterministic
           early stopping. Training-time noise is unchanged.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_class().to(DEVICE)   # seeded construction

    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    crit = nn.HuberLoss(delta=1.0)

    Xt = torch.FloatTensor(Xtr).to(DEVICE)
    Xv = torch.FloatTensor(Xva).to(DEVICE)   # used CLEAN, no noise

    best, pat, best_state = float('inf'), 0, None
    stop_epoch = epochs

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = Xt[idx[i:i+512]]
            n = torch.clamp(b + torch.randn_like(b)*noise, -10, 10)
            opt.zero_grad()
            crit(model(n), b).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), Xv).item()   # FIXED: clean validation
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


def sample_rows(X, n, seed):
    """
    Returns a random sample of n rows from X, seeded for reproducibility.

    FIX (percentile-sampling bug): the original code used X[:50000] to
    estimate percentile normalization bounds. When X is a concatenation
    of multiple domains (e.g. np.concatenate([CIC17, CIC18])), the first
    50000 rows are entirely from whichever domain was listed first —
    so the percentile bounds were fit on ONE domain's score distribution,
    not the true pooled training distribution the model was trained on.
    This affects F1/threshold calibration (tau is derived from these
    percentiles) though not AUC/ranking, since AUC is invariant to the
    monotonic normalization transform itself.

    Seeded with the same `seed` as the model's training run, so results
    stay reproducible run-to-run without reintroducing the original
    single-domain bias.
    """
    if len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx]


def load_svdd(seed, cond, exp=None):
    name = f"svdd_seed{seed}_{cond}"
    if exp:
        name += f"_{exp}"
    path = os.path.join(CKPT_DIR, f"{name}.pt")
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
        "AUC":  roc_auc_score(yt, scores),
        "PR_AUC": average_precision_score(yt, scores),
        "F1":   f1_score(yt, yp, zero_division=0),
        "Precision": precision_score(yt, yp, zero_division=0),
        "Recall":    recall_score(yt, yp, zero_division=0),
        "TPR@FPR001": float(np.interp(0.01, fpr, tpr)),
        "TPR@FPR005": float(np.interp(0.05, fpr, tpr)),
    }


# ── Data loading (subsampling logic unchanged from originals) ────
def load_all():
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
        tests[name] = (np.clip(d["X"], -10, 10).astype(np.float32),
                       d["y"].astype(int))
        log.info("  %s: %d rows", name, len(tests[name][0]))

    rng = np.random.default_rng(0)   # same fixed seed as original scripts,
                                       # for data subsampling only — not model training
    idx17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    idxus = rng.choice(len(X_usw), N_PER_ENV, replace=False)
    vi17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)
    vius  = rng.choice(len(V_usw), N_PER_ENV//4, replace=False)

    idx_s = rng.choice(len(X_c17), N_SINGLE, replace=False)
    vi_s  = rng.choice(len(V_c17), N_SINGLE//4, replace=False)

    X_single, V_single = X_c17[idx_s], V_c17[vi_s]
    X_multi  = np.concatenate([X_c17[idx17], X_c18[idx18], X_usw[idxus]])
    V_multi  = np.concatenate([V_c17[vi17],  V_c18[vi18],  V_usw[vius]])

    envs = {
        "CIC17": (X_c17[idx17], V_c17[vi17]),
        "CIC18": (X_c18[idx18], V_c18[vi18]),
        "UNSW":  (X_usw[idxus], V_usw[vius]),
    }

    # Clean diversity: single=200k CIC17, multi=100k CIC17+100k CIC18
    idx_cd_s   = rng.choice(len(X_c17), N_PER_ENV*2, replace=False)
    idx_cd_m17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx_cd_m18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    vi_cd_s    = rng.choice(len(V_c17), N_PER_ENV//2, replace=False)
    vi_cd_m17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi_cd_m18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)

    X_cd_s = X_c17[idx_cd_s]
    V_cd_s = V_c17[vi_cd_s]
    X_cd_m = np.concatenate([X_c17[idx_cd_m17], X_c18[idx_cd_m18]])
    V_cd_m = np.concatenate([V_c17[vi_cd_m17],  V_c18[vi_cd_m18]])

    return (X_single, V_single, X_multi, V_multi, tests, envs,
            X_cd_s, V_cd_s, X_cd_m, V_cd_m)


# ── Resumability helpers ─────────────────────────────────────────
# Each of these CSVs is appended to incrementally (one seed/condition's
# worth of rows at a time), not held in memory and written once at the
# end. If the script crashes, rerunning it re-reads these files and
# skips any (condition/experiment, seed) combination already present,
# so you only lose the current in-progress seed, not the whole run.
T3_PATH  = os.path.join(OUT_DIR, "table3_single_multi.csv")
T4_PATH  = os.path.join(OUT_DIR, "table4_lodo.csv")
T5_PATH  = os.path.join(OUT_DIR, "table5_extended_metrics.csv")
T6_PATH  = os.path.join(OUT_DIR, "table6_score_correlation.csv")
T7_PATH  = os.path.join(OUT_DIR, "table7_clean_diversity.csv")
STAT_PATH= os.path.join(OUT_DIR, "paired_stats.csv")


def append_rows(path, rows):
    """Append rows (list of dicts) to a CSV, writing header only if new."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=write_header, index=False)


def done_keys(path, key_cols):
    """Return the set of already-completed key tuples from a CSV, or
    an empty set if the file doesn't exist yet."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if not all(c in df.columns for c in key_cols):
        return set()
    return set(map(tuple, df[key_cols].drop_duplicates().values))


# ── Main ───────────────────────────────────────────────────────
def run():
    (X_single, V_single, X_multi, V_multi, tests, envs,
     X_cd_s, V_cd_s, X_cd_m, V_cd_m) = load_all()

    # What's already done, from any previous (possibly crashed) run
    t3_done   = done_keys(T3_PATH,  ["condition", "seed"])
    t4_done   = done_keys(T4_PATH,  ["experiment", "seed"])
    t7_done   = done_keys(T7_PATH,  ["seed"])
    stat_done = done_keys(STAT_PATH,["condition", "seed"])
    log.info("Resuming: %d Table III combos, %d Table IV combos, "
             "%d Table VII seeds already done.",
             len(t3_done), len(t4_done), len(t7_done))

    # ═══ Table III: Single vs Multi ═══
    log.info("="*60); log.info("TABLE III: Single vs Multi")
    for cond, Xtr, Xva in [("Single", X_single, V_single),
                             ("Multi",  X_multi,  V_multi)]:
        for seed in SEEDS:
            if (cond, seed) in t3_done:
                log.info("  [%s] seed %d — already done, skipping", cond, seed)
                continue
            log.info("  [%s] seed %d", cond, seed)
            t3_rows, t5_rows, t6_rows = [], [], []   # this seed's rows only
            dae, ep_d = train_ae(DAE, Xtr, Xva, seed)
            trae, ep_t = train_ae(TransformerAE, Xtr, Xva, seed)
            log.info("    stopped: DAE ep%d  TrAE ep%d", ep_d, ep_t)

            trs_d = get_ae_scores(dae, sample_rows(Xtr, 50000, seed))
            p1d, p99d = np.percentile(trs_d,1), np.percentile(trs_d,99)
            trs_t = get_ae_scores(trae, sample_rows(Xtr, 50000, seed))
            p1t, p99t = np.percentile(trs_t,1), np.percentile(trs_t,99)

            vs_d = norm(get_ae_scores(dae, Xva), p1d, p99d)
            vs_t = norm(get_ae_scores(trae, Xva), p1t, p99t)
            tau_d, tau_t = np.percentile(vs_d,95), np.percentile(vs_t,95)

            svdd_enc, svdd_c = load_svdd(seed, cond)

            for ds, (Xt, yt) in tests.items():
                sd = norm(get_ae_scores(dae, Xt), p1d, p99d)
                st = norm(get_ae_scores(trae, Xt), p1t, p99t)
                auc_d, auc_t = roc_auc_score(yt, sd), roc_auc_score(yt, st)

                se2 = (sd + st) / 2
                tau_e2 = np.percentile((vs_d+vs_t)/2, 95)
                auc_e2 = roc_auc_score(yt, se2)

                row = {"condition": cond, "seed": seed, "dataset": ds,
                       "DAE_AUC": auc_d, "TrAE_AUC": auc_t, "Ens2_AUC": auc_e2}

                if svdd_enc is not None:
                    trs_sv = get_svdd_scores(svdd_enc, svdd_c, sample_rows(Xtr, 50000, seed))
                    p1sv, p99sv = np.percentile(trs_sv,1), np.percentile(trs_sv,99)
                    vs_sv = norm(get_svdd_scores(svdd_enc, svdd_c, Xva), p1sv, p99sv)
                    tau_sv = np.percentile(vs_sv, 95)   # FIX: was missing, caused UnboundLocalError
                    ssv = norm(get_svdd_scores(svdd_enc, svdd_c, Xt), p1sv, p99sv)
                    auc_sv = roc_auc_score(yt, ssv)
                    se3 = (sd + st + ssv) / 3
                    tau_e3 = np.percentile((vs_d+vs_t+vs_sv)/3, 95)
                    auc_e3 = roc_auc_score(yt, se3)
                    row["SVDD_AUC"] = auc_sv
                    row["Ens3_AUC"] = auc_e3

                    # Table VI correlation (Multi condition, UNSW only, matches original scope)
                    if cond == "Multi" and ds == "UNSW":
                        pr_dt,_ = pearsonr(sd, st)
                        pr_ds,_ = pearsonr(sd, ssv)
                        pr_ts,_ = pearsonr(st, ssv)
                        t6_rows += [
                            {"seed":seed,"pair":"DAE-TrAE","pearson":pr_dt},
                            {"seed":seed,"pair":"DAE-SVDD","pearson":pr_ds},
                            {"seed":seed,"pair":"TrAE-SVDD","pearson":pr_ts},
                        ]

                t3_rows.append(row)

                # Table V extended metrics (Multi, UNSW only, matches original scope)
                if cond == "Multi" and ds == "UNSW":
                    t5_rows.append({**full_metrics(sd,yt,tau_d), "model":"DAE","seed":seed})
                    t5_rows.append({**full_metrics(st,yt,tau_t), "model":"TrAE","seed":seed})
                    t5_rows.append({**full_metrics(se2,yt,tau_e2), "model":"Ens2","seed":seed})
                    if svdd_enc is not None:
                        t5_rows.append({**full_metrics(ssv,yt,tau_sv), "model":"SVDD","seed":seed})
                        t5_rows.append({**full_metrics(se3,yt,tau_e3), "model":"Ens3","seed":seed})

            # Paired stats (single vs multi, DAE, UNSW) for Wilcoxon
            this_stat_rows = []
            if (cond, seed) not in stat_done:
                Xt_u, yt_u = tests["UNSW"]
                auc_u = roc_auc_score(yt_u, norm(get_ae_scores(dae, Xt_u), p1d, p99d))
                this_stat_rows.append({"seed": seed, "condition": cond, "model": "DAE", "UNSW_AUC": auc_u})

            # Persist this seed's results immediately — a crash after this
            # point only loses the NEXT seed's work, not this one.
            append_rows(T3_PATH, t3_rows)
            append_rows(T5_PATH, t5_rows)
            append_rows(T6_PATH, t6_rows)
            append_rows(STAT_PATH, this_stat_rows)
            log.info("  [%s] seed %d — saved to disk", cond, seed)

    # ═══ Table IV: LODO-1/2/3 ═══
    log.info("="*60); log.info("TABLE IV: LODO")
    lodo_configs = [
        ("LODO-1", ["CIC18","UNSW"], "CIC17"),
        ("LODO-2", ["CIC17","UNSW"], "CIC18"),
        ("LODO-3", ["CIC17","CIC18"], "UNSW"),
    ]
    for exp, train_envs, held_out in lodo_configs:
        Xtr = np.concatenate([envs[e][0] for e in train_envs])
        Xva = np.concatenate([envs[e][1] for e in train_envs])
        X_test, y_test = tests[held_out]
        log.info("  %s: train=%s held_out=%s", exp, train_envs, held_out)

        for seed in SEEDS:
            if (exp, seed) in t4_done:
                log.info("    seed %d — already done, skipping", seed)
                continue
            log.info("    seed %d", seed)
            dae, ep_d = train_ae(DAE, Xtr, Xva, seed)
            trae, ep_t = train_ae(TransformerAE, Xtr, Xva, seed)
            log.info("      stopped: DAE ep%d  TrAE ep%d", ep_d, ep_t)

            trs_d = get_ae_scores(dae, sample_rows(Xtr, 50000, seed))
            p1d, p99d = np.percentile(trs_d,1), np.percentile(trs_d,99)
            trs_t = get_ae_scores(trae, sample_rows(Xtr, 50000, seed))
            p1t, p99t = np.percentile(trs_t,1), np.percentile(trs_t,99)
            vs_d = norm(get_ae_scores(dae, Xva), p1d, p99d)
            vs_t = norm(get_ae_scores(trae, Xva), p1t, p99t)
            tau_d, tau_t = np.percentile(vs_d,95), np.percentile(vs_t,95)

            sd = norm(get_ae_scores(dae, X_test), p1d, p99d)
            st = norm(get_ae_scores(trae, X_test), p1t, p99t)
            auc_d = roc_auc_score(y_test, sd)
            auc_t = roc_auc_score(y_test, st)
            f1_d = f1_score(y_test, (sd>tau_d).astype(int), zero_division=0)
            f1_t = f1_score(y_test, (st>tau_t).astype(int), zero_division=0)

            se2 = (sd+st)/2; tau_e2 = (tau_d+tau_t)/2
            auc_e2 = roc_auc_score(y_test, se2)
            f1_e2 = f1_score(y_test, (se2>tau_e2).astype(int), zero_division=0)

            rows_this = [
                {"experiment":exp,"held_out":held_out,"model":"DAE","seed":seed,"AUC":auc_d,"F1":f1_d},
                {"experiment":exp,"held_out":held_out,"model":"TrAE","seed":seed,"AUC":auc_t,"F1":f1_t},
                {"experiment":exp,"held_out":held_out,"model":"Ens2","seed":seed,"AUC":auc_e2,"F1":f1_e2},
            ]

            svdd_enc, svdd_c = load_svdd(seed, exp, held_out)
            if svdd_enc is not None:
                trs_sv = get_svdd_scores(svdd_enc, svdd_c, sample_rows(Xtr, 50000, seed))
                p1sv, p99sv = np.percentile(trs_sv,1), np.percentile(trs_sv,99)
                vs_sv = norm(get_svdd_scores(svdd_enc, svdd_c, Xva), p1sv, p99sv)
                tau_sv = np.percentile(vs_sv, 95)
                ssv = norm(get_svdd_scores(svdd_enc, svdd_c, X_test), p1sv, p99sv)
                auc_sv = roc_auc_score(y_test, ssv)
                f1_sv = f1_score(y_test, (ssv>tau_sv).astype(int), zero_division=0)
                se3 = (sd+st+ssv)/3; tau_e3 = (tau_d+tau_t+tau_sv)/3
                auc_e3 = roc_auc_score(y_test, se3)
                f1_e3 = f1_score(y_test, (se3>tau_e3).astype(int), zero_division=0)
                rows_this.append({"experiment":exp,"held_out":held_out,"model":"SVDD","seed":seed,"AUC":auc_sv,"F1":f1_sv})
                rows_this.append({"experiment":exp,"held_out":held_out,"model":"Ens3","seed":seed,"AUC":auc_e3,"F1":f1_e3})
            else:
                log.warning("      No SVDD checkpoint for %s/%s seed%d -> Ens3 skipped", exp, held_out, seed)

            append_rows(T4_PATH, rows_this)
            log.info("      DAE=%.4f TrAE=%.4f Ens2=%.4f — saved to disk", auc_d, auc_t, auc_e2)

    # ═══ Table VII: Clean diversity ═══
    log.info("="*60); log.info("TABLE VII: Clean diversity")
    Xt_u, yt_u = tests["UNSW"]
    for seed in SEEDS:
        if (seed,) in t7_done:
            log.info("  seed %d — already done, skipping", seed)
            continue
        log.info("  seed %d", seed)
        dae_s, ep_s = train_ae(DAE, X_cd_s, V_cd_s, seed)
        dae_m, ep_m = train_ae(DAE, X_cd_m, V_cd_m, seed)
        log.info("    stopped: single ep%d  multi ep%d", ep_s, ep_m)

        trs_s = get_ae_scores(dae_s, sample_rows(X_cd_s, 50000, seed))
        p1s, p99s = np.percentile(trs_s,1), np.percentile(trs_s,99)
        auc_s = roc_auc_score(yt_u, norm(get_ae_scores(dae_s, Xt_u), p1s, p99s))

        trs_m = get_ae_scores(dae_m, sample_rows(X_cd_m, 50000, seed))
        p1m, p99m = np.percentile(trs_m,1), np.percentile(trs_m,99)
        auc_m = roc_auc_score(yt_u, norm(get_ae_scores(dae_m, Xt_u), p1m, p99m))

        row = {"seed": seed, "DAE_single_200k_CIC17": auc_s,
               "DAE_multi_100k_CIC17_CIC18": auc_m, "diff": auc_m - auc_s}
        append_rows(T7_PATH, [row])
        log.info("    single=%.4f multi=%.4f diff=%+.4f — saved to disk",
                  auc_s, auc_m, auc_m-auc_s)

    # ═══ Wilcoxon (single vs multi, DAE, UNSW) — read final state from disk ═══
    if os.path.exists(STAT_PATH):
        sdf = pd.read_csv(STAT_PATH)
        s_vals = sdf[sdf["condition"]=="Single"].sort_values("seed")["UNSW_AUC"].values
        m_vals = sdf[sdf["condition"]=="Multi"].sort_values("seed")["UNSW_AUC"].values
        if len(s_vals) == len(m_vals) == len(SEEDS):
            stat, p = wilcoxon(m_vals, s_vals, alternative='greater')
            log.info("Wilcoxon (multi > single, DAE UNSW): W=%.0f p=%.4f", stat, p)
        else:
            log.info("Wilcoxon skipped: paired_stats.csv doesn't yet have all %d seeds "
                      "for both conditions.", len(SEEDS))

    log.info("All done. Results in %s", OUT_DIR)


if __name__ == "__main__":
    log.info("Paper 3 — Consolidated fixed rerun (DAE + TrAE)")
    run()
