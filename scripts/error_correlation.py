

"""
error_correlation.py
====================
Computes Pearson and Spearman correlation
between DAE and TrAE anomaly scores on
all test sets under multi-environment training.

Low correlation -> ensemble is justified
High correlation -> ensemble adds little

Output:
  paper3_ensemble/results/error_correlation.csv

Expected time: ~90 minutes (trains both models
once per seed under multi-env condition)
"""
import os, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP    = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
OUT_DIR  = r"C:/MLProject/paper3_ensemble/results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS    = [42, 123, 456, 789, 999]
NOISE    = 0.1
N_PER_ENV = 100_000


class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(18,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,8),  nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(8,16),  nn.ReLU(),
            nn.Linear(16,32), nn.ReLU(),
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,18))
    def forward(self, x):
        return self.dec(self.enc(x))


class TransformerAE(nn.Module):
    def __init__(self, n_feat=18, d_model=64,
                 n_heads=4, d_ff=256,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb    = nn.Embedding(n_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1))
        self.n_feat = n_feat

    def forward(self, x):
        tokens  = self.input_proj(x.unsqueeze(-1))
        pos     = torch.arange(self.n_feat,
                               device=x.device)
        tokens  = tokens + self.pos_emb(pos).unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.decoder(encoded).squeeze(-1)


def train_model(model, Xtr, Xva, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt  = torch.optim.Adam(model.parameters(),
                            lr=1e-3)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(
               opt, patience=5, factor=0.5)
    crit = nn.HuberLoss(delta=1.0)
    Xt   = torch.FloatTensor(Xtr)
    Xv   = torch.FloatTensor(Xva)
    best, pat, bs = float('inf'), 0, None

    for ep in range(100):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = Xt[idx[i:i+512]]
            n = torch.clamp(
                b + torch.randn_like(b)*NOISE,
                -10, 10)
            opt.zero_grad()
            crit(model(n), b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            nv = torch.clamp(
                Xv + torch.randn_like(Xv)*NOISE,
                -10, 10)
            vl = crit(model(nv), Xv).item()
        sch.step(vl)
        if vl < best:
            best = vl
            bs   = {k: v.clone()
                    for k, v in model.state_dict().items()}
            pat  = 0
        else:
            pat += 1
            if pat >= 10:
                log.info("  Early stop ep %d", ep+1)
                break
    model.load_state_dict(bs)
    return model


def get_scores(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            out.append(
                ((b - model(b))**2)
                .mean(dim=1).numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip(
        (s - p1) / (p99 - p1 + 1e-8), 0, 1)


def run():
    log.info("Error correlation analysis")
    log.info("Seeds: %s", SEEDS)

    # Load multi-env training data
    d     = np.load(os.path.join(DEDUP,
                                 "train_dedup.npz"))
    X_all = np.clip(d["X_train"],
                    -10, 10).astype(np.float32)
    n     = len(X_all) // 3

    d     = np.load(os.path.join(DEDUP,
                                 "val_dedup.npz"))
    V_all = np.clip(d["X_val"],
                    -10, 10).astype(np.float32)
    nv    = len(V_all) // 3

    rng   = np.random.default_rng(0)
    idx17 = rng.choice(n, N_PER_ENV, replace=False)
    idx18 = rng.choice(n, N_PER_ENV, replace=False)
    idxus = rng.choice(n, N_PER_ENV, replace=False)
    vi17  = rng.choice(nv, N_PER_ENV//4,
                       replace=False)
    vi18  = rng.choice(nv, N_PER_ENV//4,
                       replace=False)
    vius  = rng.choice(nv, N_PER_ENV//4,
                       replace=False)

    Xtr = np.concatenate([X_all[:n][idx17],
                          X_all[n:2*n][idx18],
                          X_all[2*n:][idxus]])
    Xva = np.concatenate([V_all[:nv][vi17],
                          V_all[nv:2*nv][vi18],
                          V_all[2*nv:][vius]])
    log.info("Multi train: %d  val: %d",
             len(Xtr), len(Xva))

    # Load test sets
    tests = {}
    for name, fname in [
        ("CIC17", "test_cic17_dedup.npz"),
        ("CIC18", "test_cic18_dedup.npz"),
        ("UNSW",  "test_unsw_dedup.npz"),
    ]:
        d = np.load(os.path.join(DEDUP, fname))
        tests[name] = (
            np.clip(d["X"], -10, 10).astype(
                np.float32),
            d["y"].astype(int))
        log.info("  %s: %d rows", name,
                 len(tests[name][0]))

    rows = []

    for seed in SEEDS:
        log.info("="*50)
        log.info("Seed %d...", seed)
        t0 = time.time()

        # Train DAE
        dae = train_model(DAE(), Xtr, Xva, seed)
        trs = get_scores(dae, Xtr[:50000])
        p1d  = float(np.percentile(trs, 1))
        p99d = float(np.percentile(trs, 99))

        # Train TrAE
        trae = train_model(
            TransformerAE(), Xtr, Xva, seed)
        trs  = get_scores(trae, Xtr[:50000])
        p1t  = float(np.percentile(trs, 1))
        p99t = float(np.percentile(trs, 99))

        # Compute correlations on each test set
        for name, (Xt, yt) in tests.items():
            s_dae  = norm(get_scores(dae, Xt),
                          p1d, p99d)
            s_trae = norm(get_scores(trae, Xt),
                          p1t, p99t)

            # Overall correlation (all flows)
            pr_all, _  = pearsonr(s_dae, s_trae)
            sp_all, _  = spearmanr(s_dae, s_trae)

            # Benign-only correlation
            s_d_b = s_dae[yt == 0]
            s_t_b = s_trae[yt == 0]
            pr_b, _ = pearsonr(s_d_b, s_t_b)

            # Attack-only correlation
            s_d_a = s_dae[yt == 1]
            s_t_a = s_trae[yt == 1]
            pr_a, _ = pearsonr(s_d_a, s_t_a)

            # AUC for reference
            auc_d = roc_auc_score(yt, s_dae)
            auc_t = roc_auc_score(yt, s_trae)
            auc_e = roc_auc_score(
                yt, (s_dae + s_trae) / 2)

            log.info(
                "  %s: r=%.3f (benign=%.3f, "
                "attack=%.3f) DAE=%.3f "
                "TrAE=%.3f Ens=%.3f",
                name, pr_all, pr_b, pr_a,
                auc_d, auc_t, auc_e)

            rows.append({
                "seed":     seed,
                "dataset":  name,
                "pearson_all":    round(pr_all, 4),
                "pearson_benign": round(pr_b, 4),
                "pearson_attack": round(pr_a, 4),
                "spearman_all":   round(sp_all, 4),
                "DAE_AUC":  round(auc_d, 4),
                "TrAE_AUC": round(auc_t, 4),
                "Ens_AUC":  round(auc_e, 4),
            })

        log.info("Seed %d done %.1f min",
                 seed, (time.time()-t0)/60)

    df  = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR,
                       "error_correlation.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)

    # Print summary
    print("\n" + "="*65)
    print("ERROR CORRELATION SUMMARY "
          "(mean across 5 seeds)")
    print("="*65)
    print(f"{'Dataset':<8} {'Pearson(all)':>14} "
          f"{'Pearson(benign)':>16} "
          f"{'Pearson(attack)':>16}")
    print("-"*58)
    for name in ["CIC17", "CIC18", "UNSW"]:
        sub = df[df["dataset"] == name]
        pr  = sub["pearson_all"].mean()
        prb = sub["pearson_benign"].mean()
        pra = sub["pearson_attack"].mean()
        print(f"{name:<8} {pr:>14.3f} "
              f"{prb:>16.3f} {pra:>16.3f}")

    print()
    print("Interpretation:")
    print("  r < 0.7 -> low correlation "
          "-> ensemble justified")
    print("  r > 0.9 -> high correlation "
          "-> ensemble adds little")


if __name__ == "__main__":
    run()
