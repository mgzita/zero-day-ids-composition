"""
diagnostic_raw_scores_seed456.py
==================================
Second data point for the benign/attack reversal check. Retrains DAE
for the CLEAN-DIVERSITY SINGLE-SOURCE condition (200k CIC17 only,
no CIC18), seed 456 — the seed that gave AUC=0.1399 in the real
Table VII run — and inspects raw reconstruction error split by label
on UNSW.

Reuses the exact same np.random.default_rng(0) call SEQUENCE as
paper3_consolidated.py's load_all(), including the unused
idx17/idx18/idxus/vi17/vi18/vius/idx_s/vi_s draws, idx_cd_m17,
idx_cd_m18, vi_cd_m17, vi_cd_m18 draws — even though this diagnostic
doesn't need most of those subsamples, the RNG must advance through
the same sequence of calls for idx_cd_s and vi_cd_s to land on the
SAME 200k-flow subsample the real run used. Skipping any call in
that sequence would silently produce a different subsample and
defeat the point of reproducing this specific seed's result.

Expected time: ~15-30 min (DAE only, 200k flows — smaller/faster
than the LODO conditions).
"""

import os, logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
SEED  = 456
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
    """Same fixed logic as paper3_consolidated.py: seeded init, clean
    validation loss."""
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

    # ── Replay the EXACT same rng call sequence as load_all() in
    # paper3_consolidated.py, up through idx_cd_s / vi_cd_s, so this
    # reproduces the identical 200k-flow single-source subsample the
    # real run used. Every intermediate call must stay, even though
    # its result isn't used below — it's needed to advance the RNG
    # state correctly.
    rng = np.random.default_rng(0)
    _idx17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    _idx18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    _idxus = rng.choice(len(X_usw), N_PER_ENV, replace=False)
    _vi17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    _vi18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)
    _vius  = rng.choice(len(V_usw), N_PER_ENV//4, replace=False)

    _idx_s = rng.choice(len(X_c17), N_SINGLE, replace=False)
    _vi_s  = rng.choice(len(V_c17), N_SINGLE//4, replace=False)

    idx_cd_s   = rng.choice(len(X_c17), N_PER_ENV*2, replace=False)   # <-- the one we need
    _idx_cd_m17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    _idx_cd_m18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    vi_cd_s    = rng.choice(len(V_c17), N_PER_ENV//2, replace=False)   # <-- and this one
    _vi_cd_m17 = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    _vi_cd_m18 = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)

    X_cd_s = X_c17[idx_cd_s]
    V_cd_s = V_c17[vi_cd_s]
    log.info("Clean-diversity single-source: %d train, %d val (CIC17 only)",
              len(X_cd_s), len(V_cd_s))

    log.info("Training DAE (clean-diversity single-source, seed %d)...", SEED)
    dae, stop_ep = train_ae(DAE, X_cd_s, V_cd_s, SEED)
    log.info("Stopped at epoch %d", stop_ep)

    raw_scores = get_ae_scores(dae, X_test_unsw)
    auc = roc_auc_score(y_test_unsw, raw_scores)
    log.info("Recomputed AUC (should be near 0.1399 if reproducible): %.4f", auc)

    benign_scores = raw_scores[y_test_unsw == 0]
    attack_scores = raw_scores[y_test_unsw == 1]

    log.info("="*55)
    log.info("RAW reconstruction error (pre-normalization):")
    log.info("  Benign (n=%d): mean=%.6f  median=%.6f  std=%.6f",
              len(benign_scores), benign_scores.mean(), np.median(benign_scores), benign_scores.std())
    log.info("  Attack (n=%d): mean=%.6f  median=%.6f  std=%.6f",
              len(attack_scores), attack_scores.mean(), np.median(attack_scores), attack_scores.std())
    log.info("="*55)

    if attack_scores.mean() < benign_scores.mean() and auc < 0.5:
        log.info("DIAGNOSIS: Attacks reconstruct BETTER (lower error) than benign.")
        log.info("This MATCHES the seed-123/LODO-3 pattern — second confirmation")
        log.info("of a real, reproducible reversal, not a one-off artifact.")
    elif abs(auc - 0.1399) > 0.15:
        log.info("DIAGNOSIS: Could not reproduce the original 0.1399 AUC closely —")
        log.info("training may not be fully deterministic on this machine/setup.")
    else:
        log.info("DIAGNOSIS: Reproduced the low AUC, but benign/attack raw error")
        log.info("means don't show a clean reversal here — pattern may not be")
        log.info("universal across conditions/seeds. Worth noting the boundary.")


if __name__ == "__main__":
    main()
