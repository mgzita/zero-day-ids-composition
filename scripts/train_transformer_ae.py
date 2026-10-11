"""
train_transformer_ae.py
=======================
Paper 3: Transformer Autoencoder for unsupervised IDS.

Trains TransformerAE under same protocol as Paper 2 DAE:
  - Same dedup npz files
  - Same IQR scaler
  - Same 5 seeds
  - Same threshold (p95)
  - Same evaluation metrics

Conditions:
  A. Single-environment (CIC17 only, 536,307 flows)
  B. Multi-environment full (1,608,922 flows)

Evaluation:
  LODO-1: train CIC18+UNSW, test CIC17
  LODO-2: train CIC17+UNSW, test CIC18
  LODO-3: train CIC17+CIC18, test UNSW
  BoT-IoT: fully unseen (DR only)

Output:
  paper3_ensemble/results/transformer_ae_results.csv
  paper3_ensemble/results/transformer_lodo_results.csv

Expected time: ~8 hours
"""
import os, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────
DEDUP   = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
OUT_DIR = r"C:/MLProject/paper3_ensemble/results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 999]
NOISE = 0.1   # same as DAE


# ── Transformer Autoencoder ────────────────────────────────
class TransformerAE(nn.Module):
    """
    Transformer Autoencoder for tabular anomaly detection.

    Each of the 18 features is treated as a token.
    Encoder: linear embedding + 2 Transformer encoder layers.
    Decoder: linear projection back to 18 dims.
    Anomaly score: per-flow mean squared reconstruction error.
    """
    def __init__(self,
                 n_features=18,
                 d_model=64,
                 n_heads=4,
                 d_ff=256,
                 n_layers=2,
                 dropout=0.1):
        super().__init__()

        # Embed each scalar feature to d_model dimensions
        self.input_proj = nn.Linear(1, d_model)

        # Learned positional encoding (features are not sequential)
        self.pos_emb = nn.Embedding(n_features, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer,
                                             num_layers=n_layers)

        # Decoder: project back to scalar per feature
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1),
        )

        self.n_features = n_features

    def forward(self, x):
        # x: (batch, 18)
        B = x.shape[0]

        # Embed: (batch, 18, d_model)
        tokens = self.input_proj(x.unsqueeze(-1))

        # Add positional embeddings
        pos = torch.arange(self.n_features, device=x.device)
        tokens = tokens + self.pos_emb(pos).unsqueeze(0)

        # Encode: (batch, 18, d_model)
        encoded = self.encoder(tokens)

        # Decode: (batch, 18)
        recon = self.decoder(encoded).squeeze(-1)
        return recon


# ── Training ───────────────────────────────────────────────
def train_model(X_train, X_val, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TransformerAE()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=5, factor=0.5)
    crit  = nn.HuberLoss(delta=1.0)

    Xt = torch.FloatTensor(X_train)
    Xv = torch.FloatTensor(X_val)

    best, pat, best_state = float('inf'), 0, None

    for ep in range(100):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = Xt[idx[i:i+256]]
            # Gaussian noise injection (same as DAE)
            n = torch.clamp(b + torch.randn_like(b) * NOISE,
                            -10, 10)
            opt.zero_grad()
            crit(model(n), b).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            nv = torch.clamp(
                Xv + torch.randn_like(Xv) * NOISE, -10, 10)
            vl = crit(model(nv), Xv).item()

        sched.step(vl)
        if vl < best:
            best = vl
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 10:
                log.info("    Early stop ep %d", ep + 1)
                break

    model.load_state_dict(best_state)
    return model


# ── Scoring ────────────────────────────────────────────────
def get_scores(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            out.append(((b - model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


# ── Load data ──────────────────────────────────────────────
def load_data():
    log.info("Loading dedup data...")
    d = np.load(os.path.join(DEDUP, "train_dedup.npz"))
    X_full = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n = len(X_full) // 3
    X_single = X_full[:n]  # CIC17 only

    d = np.load(os.path.join(DEDUP, "val_dedup.npz"))
    V_full = np.clip(d["X_val"], -10, 10).astype(np.float32)
    nv = len(V_full) // 3
    V_single = V_full[:nv]

    tests = {}
    for name, fname in [
        ("CIC17", "test_cic17_dedup.npz"),
        ("CIC18", "test_cic18_dedup.npz"),
        ("UNSW",  "test_unsw_dedup.npz"),
        ("BoT",   "test_botiot_dedup.npz"),
    ]:
        d = np.load(os.path.join(DEDUP, fname))
        tests[name] = (
            np.clip(d["X"], -10, 10).astype(np.float32),
            d["y"].astype(int))
        log.info("  %s: %d rows", name, len(tests[name][0]))

    log.info("Single: %d  Multi: %d  Val_single: %d  Val_multi: %d",
             len(X_single), len(X_full),
             len(V_single), len(V_full))

    return X_single, X_full, V_single, V_full, tests


# ── Main experiment: single vs multi ──────────────────────
def run_main(X_single, X_full, V_single, V_full, tests):
    rows = []

    for cond, Xtr, Xva in [
        ("TrAE-Single", X_single, V_single),
        ("TrAE-Multi",  X_full,   V_full),
    ]:
        log.info("="*55)
        log.info("Condition: %s", cond)

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            model  = train_model(Xtr, Xva, seed)
            tr_s   = get_scores(model, Xtr[:50000])
            p1     = float(np.percentile(tr_s, 1))
            p99    = float(np.percentile(tr_s, 99))
            val_s  = norm(get_scores(model, Xva), p1, p99)
            tau    = float(np.percentile(val_s, 95))

            row = {"condition": cond, "seed": seed,
                   "p1": p1, "p99": p99, "tau": tau}

            for name, (Xt, yt) in tests.items():
                s = norm(get_scores(model, Xt), p1, p99)
                if name == "BoT":
                    row["BoT_DR"] = float((s > tau).mean())
                    log.info("    BoT: DR=%.4f", row["BoT_DR"])
                else:
                    auc = roc_auc_score(yt, s)
                    row[f"{name}_AUC"] = auc
                    log.info("    %s: AUC=%.4f", name, auc)

            log.info("  Seed %d done %.1f min",
                     seed, (time.time()-t0)/60)
            rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "transformer_ae_results.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)
    return df


# ── LODO experiments ───────────────────────────────────────
def run_lodo(X_full, V_full, tests):
    n = len(X_full) // 3
    nv = len(V_full) // 3

    # Environment slices (same order as Paper 2)
    envs = {
        "CIC17": (X_full[:n],       V_full[:nv]),
        "CIC18": (X_full[n:2*n],    V_full[nv:2*nv]),
        "UNSW":  (X_full[2*n:],     V_full[2*nv:]),
    }

    lodo_configs = [
        ("LODO-1", ["CIC18","UNSW"], "CIC17"),
        ("LODO-2", ["CIC17","UNSW"], "CIC18"),
        ("LODO-3", ["CIC17","CIC18"],"UNSW"),
    ]

    rows = []

    for exp_name, train_envs, held_out in lodo_configs:
        log.info("="*55)
        log.info("%s (train: %s, held-out: %s)",
                 exp_name, "+".join(train_envs), held_out)

        # Build training data from selected environments
        Xtr = np.concatenate([envs[e][0] for e in train_envs])
        Xva = np.concatenate([envs[e][1] for e in train_envs])
        log.info("  Train: %d  Val: %d", len(Xtr), len(Xva))

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            model = train_model(Xtr, Xva, seed)
            tr_s  = get_scores(model, Xtr[:50000])
            p1    = float(np.percentile(tr_s, 1))
            p99   = float(np.percentile(tr_s, 99))
            val_s = norm(get_scores(model, Xva), p1, p99)
            tau   = float(np.percentile(val_s, 95))

            row = {"experiment": exp_name,
                   "held_out": held_out,
                   "seed": seed}

            # Score all test sets
            for name, (Xt, yt) in tests.items():
                if name == "BoT":
                    continue
                s   = norm(get_scores(model, Xt), p1, p99)
                auc = roc_auc_score(yt, s)
                row[f"{name}_AUC"] = auc
                if name == held_out:
                    yp  = (s > tau).astype(int)
                    f1  = f1_score(yt, yp, zero_division=0)
                    row["held_out_F1"] = f1
                    log.info("    %s (held-out): AUC=%.4f F1=%.4f",
                             name, auc, f1)
                else:
                    log.info("    %s: AUC=%.4f", name, auc)

            log.info("  Done %.1f min", (time.time()-t0)/60)
            rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "transformer_lodo_results.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)
    return df


# ── Summary ────────────────────────────────────────────────
def print_summary(df_main, df_lodo):
    print("\n" + "="*65)
    print("TRANSFORMER AE RESULTS (mean ± pop std, 5 seeds)")
    print("="*65)

    for cond in ["TrAE-Single", "TrAE-Multi"]:
        sub = df_main[df_main["condition"]==cond]
        print(f"\n{cond}:")
        for col in ["CIC17_AUC","CIC18_AUC","UNSW_AUC","BoT_DR"]:
            if col in sub:
                v = sub[col].values
                print(f"  {col}: {v.mean():.3f}±{v.std(ddof=0):.3f}")

    # Diversity effect
    s = df_main[df_main["condition"]=="TrAE-Single"]["UNSW_AUC"].mean()
    m = df_main[df_main["condition"]=="TrAE-Multi"]["UNSW_AUC"].mean()
    print(f"\nUNSW diversity effect: {s:.3f} -> {m:.3f} ({m-s:+.3f})")

    print("\n" + "="*65)
    print("LODO RESULTS (mean ± pop std, 5 seeds)")
    print("="*65)
    for exp in ["LODO-1","LODO-2","LODO-3"]:
        sub = df_lodo[df_lodo["experiment"]==exp]
        ho  = sub["held_out"].iloc[0]
        f1  = sub["held_out_F1"].values
        auc = sub[f"{ho}_AUC"].values
        print(f"  {exp} (held-out {ho}):")
        print(f"    AUC={auc.mean():.3f}±{auc.std(ddof=0):.3f}"
              f"  F1={f1.mean():.3f}±{f1.std(ddof=0):.3f}")


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Paper 3 — Transformer AE experiments")
    log.info("Output: %s", OUT_DIR)

    X_single, X_full, V_single, V_full, tests = load_data()

    log.info("Running main experiments (single vs multi)...")
    df_main = run_main(X_single, X_full, V_single, V_full, tests)

    log.info("Running LODO experiments...")
    df_lodo = run_lodo(X_full, V_full, tests)

    print_summary(df_main, df_lodo)
    log.info("All done.")
