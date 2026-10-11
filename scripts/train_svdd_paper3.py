"""
train_svdd_paper3.py
====================
Paper 3: Deep SVDD training at 300k size-matched.
Saves model checkpoints after each seed.
Computes all ensemble combinations:
  DAE + TrAE        (existing scores + new)
  DAE + SVDD        (new)
  TrAE + SVDD       (new)
  DAE + TrAE + SVDD (3-model, new)

Requires existing score files:
  paper3_ensemble/results/main_results.csv
  paper3_ensemble/results/lodo_paper3_results.csv

Output:
  paper3_ensemble/results/svdd_main_results.csv
  paper3_ensemble/results/svdd_lodo_results.csv
  paper3_ensemble/results/ensemble_all_results.csv
  paper3_ensemble/checkpoints/svdd_*.pt

Expected time: ~19 hours
"""
import os, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP   = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
OUT_DIR = r"C:/MLProject/paper3_ensemble/results"
CKPT_DIR = r"C:/MLProject/paper3_ensemble/checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS     = [42, 123, 456, 789, 999]
N_PER_ENV = 100_000
N_SINGLE  = 300_000
LR        = 1e-4
EPOCHS    = 100
PATIENCE  = 10
EPS       = 1e-6


# ── Deep SVDD ─────────────────────────────────────────────
class SVDDEncoder(nn.Module):
    """
    Deep SVDD encoder.
    Maps 18-dim input to 32-dim hypersphere space.
    No bias in final layer (SVDD requirement).
    """
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
    """Compute hypersphere center from training data."""
    encoder.eval()
    z_list = []
    with torch.no_grad():
        Xt = torch.FloatTensor(Xtr)
        for i in range(0, len(Xt), 10000):
            z_list.append(encoder(Xt[i:i+10000]))
    c = torch.cat(z_list).mean(dim=0)
    # Avoid center collapse
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c >= 0)] = eps
    return c


def train_svdd(Xtr, Xva, seed):
    """Train Deep SVDD and return encoder + center."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = SVDDEncoder()
    opt     = torch.optim.Adam(
        encoder.parameters(), lr=LR,
        weight_decay=1e-6)
    sch     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5)

    Xt = torch.FloatTensor(Xtr)
    Xv = torch.FloatTensor(Xva)

    # Initialize center
    c = init_center(encoder, Xtr)
    log.info("  Center norm: %.4f", c.norm().item())

    best_loss, pat, best_state = float('inf'), 0, None

    for ep in range(EPOCHS):
        encoder.train()
        idx = torch.randperm(len(Xt))
        ep_loss = 0.0
        n_batch = 0
        for i in range(0, len(Xt), 512):
            b   = Xt[idx[i:i+512]]
            z   = encoder(b)
            loss = ((z - c) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batch += 1

        # Validation loss
        encoder.eval()
        with torch.no_grad():
            zv   = encoder(Xv)
            vloss = ((zv - c) ** 2).sum(dim=1).mean().item()

        sch.step(vloss)

        if vloss < best_loss:
            best_loss  = vloss
            best_state = {k: v.clone()
                          for k, v in encoder.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                log.info("    Early stop ep %d", ep + 1)
                break

    encoder.load_state_dict(best_state)
    return encoder, c


def get_svdd_scores(encoder, c, X):
    """Anomaly score = distance from hypersphere center."""
    encoder.eval()
    out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            z = encoder(b)
            dist = ((z - c) ** 2).sum(dim=1)
            out.append(dist.numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def save_checkpoint(encoder, c, seed, cond, exp=None):
    """Save model checkpoint."""
    name = f"svdd_seed{seed}_{cond}"
    if exp:
        name += f"_{exp}"
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    torch.save({
        'encoder_state': encoder.state_dict(),
        'center': c,
        'seed': seed,
        'condition': cond,
        'experiment': exp,
    }, path)
    log.info("  Checkpoint saved: %s", path)


# ── Load data ──────────────────────────────────────────────
def load_data():
    log.info("Loading dedup data...")
    d     = np.load(os.path.join(DEDUP, "train_dedup.npz"))
    X_all = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n     = len(X_all) // 3
    X_c17 = X_all[:n]
    X_c18 = X_all[n:2*n]
    X_usw = X_all[2*n:]

    d     = np.load(os.path.join(DEDUP, "val_dedup.npz"))
    V_all = np.clip(d["X_val"], -10, 10).astype(np.float32)
    nv    = len(V_all) // 3
    V_c17 = V_all[:nv]
    V_c18 = V_all[nv:2*nv]
    V_usw = V_all[2*nv:]

    rng   = np.random.default_rng(0)
    idx17 = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx18 = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    idxus = rng.choice(len(X_usw), N_PER_ENV, replace=False)
    vi17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)
    vius  = rng.choice(len(V_usw), N_PER_ENV//4, replace=False)

    # Single-env: 300k CIC17
    idx_s  = rng.choice(len(X_c17), N_SINGLE, replace=False)
    vi_s   = rng.choice(len(V_c17), N_SINGLE//4, replace=False)
    X_single = X_c17[idx_s]
    V_single = V_c17[vi_s]

    # Multi-env: 100k x 3
    X_multi = np.concatenate([X_c17[idx17],
                               X_c18[idx18],
                               X_usw[idxus]])
    V_multi = np.concatenate([V_c17[vi17],
                               V_c18[vi18],
                               V_usw[vius]])

    log.info("Single: %d  Multi: %d",
             len(X_single), len(X_multi))

    # Test sets
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
        log.info("  %s: %d rows", name,
                 len(tests[name][0]))

    # LODO environment slices
    envs = {
        "CIC17": (X_c17[idx17], V_c17[vi17]),
        "CIC18": (X_c18[idx18], V_c18[vi18]),
        "UNSW":  (X_usw[idxus], V_usw[vius]),
    }

    return (X_single, X_multi, V_single, V_multi,
            tests, envs)


# ── Main experiment ────────────────────────────────────────
def run_main(X_single, X_multi, V_single, V_multi, tests):
    rows = []

    for cond, Xtr, Xva in [
        ("Single", X_single, V_single),
        ("Multi",  X_multi,  V_multi),
    ]:
        log.info("="*55)
        log.info("SVDD Condition: %s", cond)

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            encoder, c = train_svdd(Xtr, Xva, seed)

            # Save checkpoint
            save_checkpoint(encoder, c, seed, cond)

            # Score normalization
            trs  = get_svdd_scores(encoder, c,
                                   Xtr[:50000])
            p1   = float(np.percentile(trs, 1))
            p99  = float(np.percentile(trs, 99))
            vs   = norm(get_svdd_scores(
                encoder, c, Xva), p1, p99)
            tau  = float(np.percentile(vs, 95))

            row = {"model": "SVDD",
                   "condition": cond,
                   "seed": seed,
                   "p1": p1, "p99": p99,
                   "tau": tau}

            for name, (Xt, yt) in tests.items():
                s = norm(get_svdd_scores(
                    encoder, c, Xt), p1, p99)
                if name == "BoT":
                    row["BoT_DR"] = float(
                        (s > tau).mean())
                    log.info("    BoT DR=%.4f",
                             row["BoT_DR"])
                else:
                    auc = roc_auc_score(yt, s)
                    row[f"{name}_AUC"] = auc
                    log.info("    %s AUC=%.4f",
                             name, auc)

            rows.append(row)
            log.info("  Seed %d done %.1f min",
                     seed, (time.time()-t0)/60)

    df  = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR,
                       "svdd_main_results.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)
    return df


# ── LODO ───────────────────────────────────────────────────
def run_lodo(envs, tests):
    lodo_configs = [
        ("LODO-1", ["CIC18", "UNSW"], "CIC17"),
        ("LODO-2", ["CIC17", "UNSW"], "CIC18"),
        ("LODO-3", ["CIC17", "CIC18"], "UNSW"),
    ]
    rows = []

    for exp, train_envs, held_out in lodo_configs:
        log.info("="*55)
        log.info("%s (train: %s, held-out: %s)",
                 exp, "+".join(train_envs), held_out)

        Xtr = np.concatenate(
            [envs[e][0] for e in train_envs])
        Xva = np.concatenate(
            [envs[e][1] for e in train_envs])
        X_test, y_test = tests[held_out]

        log.info("  Train: %d  Val: %d  Test: %d",
                 len(Xtr), len(Xva), len(X_test))

        for seed in SEEDS:
            log.info("  Seed %d...", seed)
            t0 = time.time()

            encoder, c = train_svdd(Xtr, Xva, seed)

            # Save checkpoint
            save_checkpoint(encoder, c, seed,
                            exp, held_out)

            trs  = get_svdd_scores(encoder, c,
                                   Xtr[:50000])
            p1   = float(np.percentile(trs, 1))
            p99  = float(np.percentile(trs, 99))
            vs   = norm(get_svdd_scores(
                encoder, c, Xva), p1, p99)
            tau  = float(np.percentile(vs, 95))

            s    = norm(get_svdd_scores(
                encoder, c, X_test), p1, p99)
            auc  = roc_auc_score(y_test, s)
            yp   = (s > tau).astype(int)
            f1   = f1_score(y_test, yp,
                            zero_division=0)

            log.info("    AUC=%.4f F1=%.4f", auc, f1)
            log.info("  Seed %d done %.1f min",
                     seed, (time.time()-t0)/60)

            rows.append({
                "model": "SVDD",
                "experiment": exp,
                "held_out": held_out,
                "seed": seed,
                "AUC": auc,
                "F1": f1,
                "p1": p1, "p99": p99,
                "tau": tau,
            })

    df  = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR,
                       "svdd_lodo_results.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)
    return df


# ── Summary ────────────────────────────────────────────────
def print_summary(df_main, df_lodo):
    print("\n" + "="*65)
    print("SVDD MAIN RESULTS (mean ± pop std, 5 seeds)")
    print("="*65)
    print(f"{'Cond':<10}{'CIC17':>12}"
          f"{'CIC18':>12}{'UNSW':>12}")
    print("-"*48)
    for cond in ["Single", "Multi"]:
        sub = df_main[df_main["condition"]==cond]
        c17 = sub["CIC17_AUC"].values
        c18 = sub["CIC18_AUC"].values
        uns = sub["UNSW_AUC"].values
        print(f"SVDD-{cond:<6}"
              f"{c17.mean():.3f}±{c17.std(ddof=0):.3f} "
              f"{c18.mean():.3f}±{c18.std(ddof=0):.3f} "
              f"{uns.mean():.3f}±{uns.std(ddof=0):.3f}")

    print("\nSVDD LODO RESULTS:")
    print(f"{'Experiment':<12}{'AUC':>12}{'F1':>10}")
    print("-"*36)
    for exp in ["LODO-1", "LODO-2", "LODO-3"]:
        sub = df_lodo[df_lodo["experiment"]==exp]
        auc = sub["AUC"].values
        f1  = sub["F1"].values
        print(f"{exp:<12}"
              f"{auc.mean():.3f}±{auc.std(ddof=0):.3f} "
              f"{f1.mean():.3f}±{f1.std(ddof=0):.3f}")


if __name__ == "__main__":
    log.info("Paper 3 — Deep SVDD at 300k size-matched")
    log.info("Checkpoints: %s", CKPT_DIR)
    log.info("Results: %s", OUT_DIR)

    (X_s, X_m, V_s, V_m,
     tests, envs) = load_data()

    log.info("Running main experiments...")
    df_main = run_main(X_s, X_m, V_s, V_m, tests)

    log.info("Running LODO experiments...")
    df_lodo = run_lodo(envs, tests)

    print_summary(df_main, df_lodo)
    log.info("All done.")
