"""
run_missing_svdd.py
====================
Fills the two missing Deep SVDD cells in Table IV (raw AUC on UNSW-NB15):

  Table7-single : train on 200k CIC17            (UNSW excluded)
  Table7-multi  : train on 100k CIC17 + 100k CIC18 (UNSW excluded)

Reuses, verbatim, from your existing scripts:
  - SVDDEncoder, init_center, train_svdd, get_svdd_scores, norm,
    save_checkpoint          <- train_svdd_paper3.py
  - data loading, clipping, dataset thirds, and the rng(0) draw order
    for the clean-diversity subsamples <- extended_analysis.py load_all()

The rng stream is replayed in the SAME order as extended_analysis.py, so
these are the identical 200k / 100k+100k subsamples that the DAE and TrAE
Table7 runs used. (The eight earlier draws are consumed and discarded
purely to advance the stream to the right state. Do not remove them.)

Checkpoints are saved as svdd_seed{seed}_CleanSingle.pt and
svdd_seed{seed}_CleanMulti.pt, so nothing existing is overwritten.

Output (OUT_DIR): svdd_table7_raw.csv
  columns: config, seed, stop_epoch_SVDD, raw_AUC_SVDD, norm_AUC_SVDD
  `config` uses the same labels as table_lodo3_table7_raw_fixed.csv
  (Table7-single / Table7-multi), so the files can be concatenated.

Resumable: skips any (config, seed) already in the output CSV.
"""

import os, time, logging
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
CKPT_DIR = r"C:/MLProject/paper3_ensemble/checkpoints"
OUT_NAME = "svdd_table7_raw.csv"

SEEDS     = [42, 123, 456, 789, 999]
N_PER_ENV = 100_000          # "N" in extended_analysis.py
LR        = 1e-4
EPOCHS    = 100
PATIENCE  = 10
EPS       = 1e-6


# ── Deep SVDD (verbatim from train_svdd_paper3.py) ─────────
class SVDDEncoder(nn.Module):
    def __init__(self, input_dim=18, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, latent_dim, bias=False))
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.net(x)


def init_center(encoder, Xtr, eps=EPS):
    encoder.eval()
    z_list = []
    with torch.no_grad():
        Xt = torch.FloatTensor(Xtr)
        for i in range(0, len(Xt), 10000):
            z_list.append(encoder(Xt[i:i+10000]))
    c = torch.cat(z_list).mean(dim=0)
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c >= 0)] = eps
    return c


def train_svdd(Xtr, Xva, seed):
    """Returns (encoder, center, stop_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = SVDDEncoder()
    opt = torch.optim.Adam(
        encoder.parameters(), lr=LR, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5)

    Xt = torch.FloatTensor(Xtr)
    Xv = torch.FloatTensor(Xva)

    c = init_center(encoder, Xtr)
    log.info("  Center norm: %.4f", c.norm().item())

    best_loss, pat, best_state = float('inf'), 0, None
    stop_epoch = EPOCHS

    for ep in range(EPOCHS):
        encoder.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = Xt[idx[i:i+512]]
            z = encoder(b)
            loss = ((z - c) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        encoder.eval()
        with torch.no_grad():
            zv = encoder(Xv)
            vloss = ((zv - c) ** 2).sum(dim=1).mean().item()
        sch.step(vloss)

        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone()
                          for k, v in encoder.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                stop_epoch = ep + 1
                log.info("    Early stop ep %d", stop_epoch)
                break

    encoder.load_state_dict(best_state)
    return encoder, c, stop_epoch


def get_svdd_scores(encoder, c, X):
    encoder.eval()
    out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            z = encoder(Xt[i:i+10000])
            out.append(((z - c) ** 2).sum(dim=1).numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def save_checkpoint(encoder, c, seed, cond):
    path = os.path.join(CKPT_DIR, f"svdd_seed{seed}_{cond}.pt")
    torch.save({'encoder_state': encoder.state_dict(),
                'center': c, 'seed': seed,
                'condition': cond, 'experiment': None}, path)
    log.info("  Checkpoint saved: %s", path)


# ── Data (replays extended_analysis.py load_all() draw order) ─
def load_clean_diversity():
    N = N_PER_ENV
    log.info("Loading dedup data...")
    d = np.load(os.path.join(DEDUP, "train_dedup.npz"))
    X_all = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n = len(X_all) // 3
    X_c17, X_c18, X_usw = X_all[:n], X_all[n:2*n], X_all[2*n:]

    d = np.load(os.path.join(DEDUP, "val_dedup.npz"))
    V_all = np.clip(d["X_val"], -10, 10).astype(np.float32)
    nv = len(V_all) // 3
    V_c17, V_c18, V_usw = V_all[:nv], V_all[nv:2*nv], V_all[2*nv:]

    rng = np.random.default_rng(0)
    # --- draws consumed only to advance the rng stream (do not remove) ---
    rng.choice(len(X_c17), N, replace=False)        # idx17
    rng.choice(len(X_c18), N, replace=False)        # idx18
    rng.choice(len(X_usw), N, replace=False)        # idxus
    rng.choice(len(V_c17), N // 4, replace=False)   # vi17
    rng.choice(len(V_c18), N // 4, replace=False)   # vi18
    rng.choice(len(V_usw), N // 4, replace=False)   # vius
    rng.choice(len(X_c17), N * 3, replace=False)    # idx_s  (300k single)
    rng.choice(len(V_c17), N * 3 // 4, replace=False)  # vi_s
    # --- clean-diversity draws (these are the ones actually used) ---
    idx_cd_s   = rng.choice(len(X_c17), N * 2, replace=False)
    idx_cd_m17 = rng.choice(len(X_c17), N, replace=False)
    idx_cd_m18 = rng.choice(len(X_c18), N, replace=False)
    vi_cd_s    = rng.choice(len(V_c17), N // 2, replace=False)
    vi_cd_m    = rng.choice(len(V_c17), N // 4, replace=False)
    vi_cd_m2   = rng.choice(len(V_c18), N // 4, replace=False)

    X_cd_s = X_c17[idx_cd_s]
    V_cd_s = V_c17[vi_cd_s]
    X_cd_m = np.concatenate([X_c17[idx_cd_m17], X_c18[idx_cd_m18]])
    V_cd_m = np.concatenate([V_c17[vi_cd_m], V_c18[vi_cd_m2]])

    d = np.load(os.path.join(DEDUP, "test_unsw_dedup.npz"))
    X_test = np.clip(d["X"], -10, 10).astype(np.float32)
    y_test = d["y"].astype(int)
    log.info("single train=%d val=%d | multi train=%d val=%d | "
             "UNSW test=%d", len(X_cd_s), len(V_cd_s),
             len(X_cd_m), len(V_cd_m), len(y_test))
    return X_cd_s, V_cd_s, X_cd_m, V_cd_m, X_test, y_test


def already_done(out_path, config, seed):
    if not os.path.exists(out_path):
        return False
    df = pd.read_csv(out_path)
    return ((df["config"] == config) & (df["seed"] == seed)).any()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_NAME)

    X_s, V_s, X_m, V_m, X_test, y_test = load_clean_diversity()
    conditions = [
        ("Table7-single", "CleanSingle", X_s, V_s),
        ("Table7-multi",  "CleanMulti",  X_m, V_m),
    ]

    for config, ck_name, Xtr, Xva in conditions:
        for seed in SEEDS:
            if already_done(out_path, config, seed):
                log.info("%s seed %d already done, skipping", config, seed)
                continue
            log.info("%s seed %d ...", config, seed)
            t0 = time.time()

            enc, c, stop_ep = train_svdd(Xtr, Xva, seed)
            save_checkpoint(enc, c, seed, ck_name)

            raw = get_svdd_scores(enc, c, X_test)
            raw_auc = roc_auc_score(y_test, raw)

            # percentile-normalised AUC, same recipe as the main pipeline
            trs = get_svdd_scores(enc, c, Xtr[:50000])
            p1, p99 = float(np.percentile(trs, 1)), float(np.percentile(trs, 99))
            norm_auc = roc_auc_score(y_test, norm(raw, p1, p99))

            row = {"config": config, "seed": seed,
                   "stop_epoch_SVDD": stop_ep,
                   "raw_AUC_SVDD": round(float(raw_auc), 4),
                   "norm_AUC_SVDD": round(float(norm_auc), 4)}
            pd.DataFrame([row]).to_csv(
                out_path, mode='a', header=not os.path.exists(out_path),
                index=False)
            log.info("  raw=%.4f norm=%.4f stop_ep=%d (%.1f min) saved",
                     raw_auc, norm_auc, stop_ep, (time.time() - t0) / 60)

    df = pd.read_csv(out_path)
    log.info("=" * 60)
    log.info("SUMMARY (mean +/- pop std, UNSW-NB15)")
    for config in ["Table7-single", "Table7-multi"]:
        sub = df[df["config"] == config]
        if len(sub) == 0:
            continue
        r, nm = sub["raw_AUC_SVDD"].values, sub["norm_AUC_SVDD"].values
        log.info("  %-14s n=%d  raw %.3f +/- %.3f   norm %.3f +/- %.3f",
                 config, len(sub), r.mean(), r.std(ddof=0),
                 nm.mean(), nm.std(ddof=0))
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
