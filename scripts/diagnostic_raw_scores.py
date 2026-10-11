"""
diagnostic_raw_scores.py
=========================
One-seed diagnostic: retrains DAE for LODO-3 (train=CIC17+CIC18,
held-out=UNSW), seed 123 — the seed that gave AUC=0.1689 in the real
run — and inspects RAW (pre-normalization) reconstruction error
separately for UNSW benign vs. attack flows.

Question this answers: is the model genuinely reconstructing attacks
better than benign traffic (a real, reportable reversal), or is
something in the pipeline scrambling labels/scores (a bug)?

Interpretation:
  - If mean/median attack error is clearly LOWER than benign error,
    and AUC recomputed here also lands well below 0.5 — that's real.
    The model has learned CIC17/CIC18-style "normal," and UNSW benign
    traffic looks less like that than some UNSW attacks do.
  - If benign/attack raw errors look similar, or AUC recomputed here
    is NOT similarly low — something is misaligned (e.g. label/score
    order) and needs further investigation before trusting LODO-3.

Expected time: ~1-2 hours (one DAE training run, matches the
per-seed timings you've been seeing for LODO configs).
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
SEED  = 123
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

    rng = np.random.default_rng(0)   # matches original subsampling seed
    idx17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    idxus = rng.choice(len(X_usw), N_PER_ENV, replace=False)   # unused here, LODO-3 excludes UNSW
    vi17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)
    vius  = rng.choice(len(V_usw), N_PER_ENV//4, replace=False)  # unused here

    Xtr = np.concatenate([X_c17[idx17], X_c18[idx18]])
    Xva = np.concatenate([V_c17[vi17],  V_c18[vi18]])

    log.info("Training DAE (LODO-3, seed %d)...", SEED)
    dae, stop_ep = train_ae(DAE, Xtr, Xva, SEED)
    log.info("Stopped at epoch %d", stop_ep)

    raw_scores = get_ae_scores(dae, X_test_unsw)
    auc = roc_auc_score(y_test_unsw, raw_scores)
    log.info("Recomputed AUC (should be near 0.1689 if reproducible): %.4f", auc)

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
        log.info("This is a REAL reversal, not a pipeline bug — the DAE trained on")
        log.info("CIC17+CIC18 benign traffic finds some UNSW attacks MORE 'normal'")
        log.info("than UNSW's own benign traffic. Worth reporting as a finding.")
    elif abs(auc - 0.1689) > 0.15:
        log.info("DIAGNOSIS: Could not reproduce the original 0.1689 AUC closely —")
        log.info("training may not be fully deterministic on this machine/setup,")
        log.info("or the original run's data slicing differed somehow. Re-examine.")
    else:
        log.info("DIAGNOSIS: Reproduced the low AUC, but benign/attack raw error")
        log.info("means don't show a clean reversal — worth checking label/score")
        log.info("alignment (e.g. y array order vs X array order) more carefully.")


if __name__ == "__main__":
    main()
