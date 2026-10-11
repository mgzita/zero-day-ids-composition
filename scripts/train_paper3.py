"""
train_paper3.py
===============
Paper 3: DAE + TransformerAE ensemble for unsupervised IDS.

Both models trained at SAME size-matched 300k flows:
  Single-env: 300,000 CIC17 flows
  Multi-env:  100,000 per environment = 300,000 total

Conditions:
  A. Single-environment (300k CIC17)
  B. Multi-environment matched (100k x 3 envs = 300k)

Evaluation:
  Main: CIC17, CIC18, UNSW AUC + BoT DR
  LODO: LODO-1, LODO-2, LODO-3

Seeds: 42, 123, 456, 789, 999

Output:
  paper3_ensemble/results/dae_300k_results.csv
  paper3_ensemble/results/trae_300k_results.csv
  paper3_ensemble/results/ensemble_300k_results.csv
  paper3_ensemble/results/lodo_300k_results.csv

Expected time: ~10 hours total
  DAE:  5 seeds x 2 conditions x ~25 min = ~4 hours
  TrAE: 5 seeds x 2 conditions x ~40 min = ~7 hours
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

DEDUP   = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
OUT_DIR = r"C:/MLProject/paper3_ensemble/results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS    = [42, 123, 456, 789, 999]
NOISE    = 0.1
N_SINGLE = 300_000   # single-env training flows
N_PER_ENV = 100_000  # multi-env flows per environment


# ── Models ─────────────────────────────────────────────────

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
    def forward(self, x): return self.dec(self.enc(x))


class TransformerAE(nn.Module):
    def __init__(self, n_feat=18, d_model=64,
                 n_heads=4, d_ff=256, n_layers=2, dropout=0.1):
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
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Linear(d_ff, 1))
        self.n_feat = n_feat

    def forward(self, x):
        tokens  = self.input_proj(x.unsqueeze(-1))
        pos     = torch.arange(self.n_feat, device=x.device)
        tokens  = tokens + self.pos_emb(pos).unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.decoder(encoded).squeeze(-1)


# ── Training ───────────────────────────────────────────────

def train_model(model, X_train, X_val, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=5, factor=0.5)
    crit  = nn.HuberLoss(delta=1.0)
    Xt    = torch.FloatTensor(X_train)
    Xv    = torch.FloatTensor(X_val)
    best, pat, bs = float('inf'), 0, None

    for ep in range(100):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):  # larger batch = faster
            b = Xt[idx[i:i+512]]
            n = torch.clamp(b + torch.randn_like(b)*NOISE, -10, 10)
            opt.zero_grad(); crit(model(n), b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            nv = torch.clamp(Xv+torch.randn_like(Xv)*NOISE,-10,10)
            vl = crit(model(nv), Xv).item()
        sched.step(vl)
        if vl < best:
            best = vl
            bs   = {k: v.clone() for k,v in model.state_dict().items()}
            pat  = 0
        else:
            pat += 1
            if pat >= 10:
                log.info("    Early stop ep %d", ep+1); break

    model.load_state_dict(bs)
    return model


# ── Scoring ────────────────────────────────────────────────

def get_scores(model, X):
    model.eval(); out = []
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            out.append(((b - model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip((s-p1)/(p99-p1+1e-8), 0, 1)


def evaluate(model, Xtr, Xva, tests):
    """Return normalized scores and tau for all test sets."""
    tr_s = get_scores(model, Xtr[:50000])
    p1   = float(np.percentile(tr_s, 1))
    p99  = float(np.percentile(tr_s, 99))
    vs   = norm(get_scores(model, Xva), p1, p99)
    tau  = float(np.percentile(vs, 95))
    scores = {}
    for name, (Xt, _) in tests.items():
        scores[name] = norm(get_scores(model, Xt), p1, p99)
    return scores, tau


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

    # Size-matched 300k training sets
    rng = np.random.default_rng(0)  # fixed for reproducibility

    # Single-env: 300k CIC17
    idx      = rng.choice(len(X_c17), N_SINGLE, replace=False)
    X_single = X_c17[idx]
    vidx     = rng.choice(len(V_c17), N_SINGLE//4, replace=False)
    V_single = V_c17[vidx]

    # Multi-env: 100k per environment
    idx17  = rng.choice(len(X_c17), N_PER_ENV, replace=False)
    idx18  = rng.choice(len(X_c18), N_PER_ENV, replace=False)
    idxusw = rng.choice(len(X_usw), N_PER_ENV, replace=False)
    X_multi = np.concatenate([X_c17[idx17],
                               X_c18[idx18],
                               X_usw[idxusw]])
    vi17  = rng.choice(len(V_c17), N_PER_ENV//4, replace=False)
    vi18  = rng.choice(len(V_c18), N_PER_ENV//4, replace=False)
    viusw = rng.choice(len(V_usw), N_PER_ENV//4, replace=False)
    V_multi = np.concatenate([V_c17[vi17],
                               V_c18[vi18],
                               V_usw[viusw]])

    log.info("Single: %d  Multi: %d  Val_s: %d  Val_m: %d",
             len(X_single), len(X_multi),
             len(V_single), len(V_multi))

    tests = {}
    for name, fname in [
        ("CIC17","test_cic17_dedup.npz"),
        ("CIC18","test_cic18_dedup.npz"),
        ("UNSW", "test_unsw_dedup.npz"),
        ("BoT",  "test_botiot_dedup.npz"),
    ]:
        d = np.load(os.path.join(DEDUP, fname))
        tests[name] = (
            np.clip(d["X"],-10,10).astype(np.float32),
            d["y"].astype(int))
        log.info("  Test %s: %d rows", name, len(tests[name][0]))

    # LODO environment slices (100k per env for speed)
    envs = {
        "CIC17": (X_c17[idx17], V_c17[vi17]),
        "CIC18": (X_c18[idx18], V_c18[vi18]),
        "UNSW":  (X_usw[idxusw],V_usw[viusw]),
    }

    return X_single, X_multi, V_single, V_multi, tests, envs


# ── Main experiment ────────────────────────────────────────

def run_main(X_single, X_multi, V_single, V_multi, tests):
    all_rows = {"dae": [], "trae": [], "ensemble": []}

    conditions = [
        ("Single", X_single, V_single),
        ("Multi",  X_multi,  V_multi),
    ]

    for cond_name, Xtr, Xva in conditions:
        log.info("="*55)
        log.info("Condition: %s", cond_name)

        for seed in SEEDS:
            log.info("  Seed %d...", seed); t0 = time.time()

            # Train DAE
            dae  = train_model(DAE(), Xtr, Xva, seed)
            dae_s, tau_d = evaluate(dae, Xtr, Xva, tests)

            # Train TrAE
            trae  = train_model(TransformerAE(), Xtr, Xva, seed)
            trae_s, tau_t = evaluate(trae, Xtr, Xva, tests)

            for model_name, scores, tau in [
                ("DAE",      dae_s,  tau_d),
                ("TrAE",     trae_s, tau_t),
            ]:
                row = {"model": model_name,
                       "condition": cond_name,
                       "seed": seed}
                for name, (_, yt) in tests.items():
                    s = scores[name]
                    if name == "BoT":
                        row["BoT_DR"] = float((s>tau).mean())
                        log.info("    %s %s BoT DR=%.4f",
                                 model_name, name, row["BoT_DR"])
                    else:
                        auc = roc_auc_score(yt, s)
                        row[f"{name}_AUC"] = auc
                        log.info("    %s %s AUC=%.4f",
                                 model_name, name, auc)
                all_rows[model_name.lower()].append(row)

            # Ensemble: average normalized scores
            ens_row = {"model":"Ensemble",
                       "condition":cond_name, "seed":seed}
            tau_ens = (tau_d + tau_t) / 2
            for name, (_, yt) in tests.items():
                s_ens = (dae_s[name] + trae_s[name]) / 2
                if name == "BoT":
                    ens_row["BoT_DR"] = float((s_ens>tau_ens).mean())
                else:
                    auc = roc_auc_score(yt, s_ens)
                    ens_row[f"{name}_AUC"] = auc
                    log.info("    Ensemble %s AUC=%.4f", name, auc)
            all_rows["ensemble"].append(ens_row)

            log.info("  Seed %d done %.1f min",
                     seed, (time.time()-t0)/60)

    # Save
    for key, rows in all_rows.items():
        df = pd.DataFrame(rows)
        out = os.path.join(OUT_DIR, f"{key}_300k_results.csv")
        df.to_csv(out, index=False)
        log.info("Saved -> %s", out)

    return {k: pd.DataFrame(v) for k, v in all_rows.items()}


# ── LODO ───────────────────────────────────────────────────

def run_lodo(envs, tests):
    lodo_configs = [
        ("LODO-1", ["CIC18","UNSW"], "CIC17"),
        ("LODO-2", ["CIC17","UNSW"], "CIC18"),
        ("LODO-3", ["CIC17","CIC18"],"UNSW"),
    ]
    rows = []

    for exp, train_envs, held_out in lodo_configs:
        log.info("="*55)
        log.info("%s (train: %s, held-out: %s)",
                 exp, "+".join(train_envs), held_out)
        Xtr = np.concatenate([envs[e][0] for e in train_envs])
        Xva = np.concatenate([envs[e][1] for e in train_envs])

        for seed in SEEDS:
            log.info("  Seed %d...", seed); t0 = time.time()

            dae  = train_model(DAE(), Xtr, Xva, seed)
            dae_s, tau_d = evaluate(dae, Xtr, Xva, tests)

            trae  = train_model(TransformerAE(), Xtr, Xva, seed)
            trae_s, tau_t = evaluate(trae, Xtr, Xva, tests)

            for mname, scores, tau in [
                ("DAE",      dae_s,  tau_d),
                ("TrAE",     trae_s, tau_t),
                ("Ensemble", {n:(dae_s[n]+trae_s[n])/2
                              for n in dae_s},
                             (tau_d+tau_t)/2),
            ]:
                row = {"model":mname, "experiment":exp,
                       "held_out":held_out, "seed":seed}
                for name, (_, yt) in tests.items():
                    if name == "BoT": continue
                    s   = scores[name]
                    auc = roc_auc_score(yt, s)
                    row[f"{name}_AUC"] = auc
                    if name == held_out:
                        yp  = (s > tau).astype(int)
                        row["held_out_F1"] = f1_score(
                            yt, yp, zero_division=0)
                        log.info("    %s %s(held-out) AUC=%.4f F1=%.4f",
                                 mname, name, auc, row["held_out_F1"])
                    else:
                        log.info("    %s %s AUC=%.4f",
                                 mname, name, auc)
                rows.append(row)

            log.info("  Seed %d done %.1f min",
                     seed, (time.time()-t0)/60)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR,"lodo_300k_results.csv"),
              index=False)
    log.info("Saved lodo_300k_results.csv")
    return df


# ── Summary ────────────────────────────────────────────────

def print_summary(dfs, df_lodo):
    print("\n" + "="*65)
    print("MAIN RESULTS (mean ± pop std, 5 seeds, 300k flows)")
    print("="*65)
    print(f"{'Model':<12}{'Cond':<10}"
          f"{'CIC17':>8}{'CIC18':>8}{'UNSW':>8}{'BoT DR':>8}")
    print("-"*55)
    for mname in ["DAE","TrAE","Ensemble"]:
        key = mname.lower().replace("ensemble","ensemble")
        df  = dfs.get(key, dfs.get(mname.lower()))
        if df is None: continue
        for cond in ["Single","Multi"]:
            sub = df[df["condition"]==cond]
            c17 = sub["CIC17_AUC"].mean() if "CIC17_AUC" in sub else 0
            c18 = sub["CIC18_AUC"].mean() if "CIC18_AUC" in sub else 0
            uns = sub["UNSW_AUC"].mean()  if "UNSW_AUC"  in sub else 0
            bot = sub["BoT_DR"].mean()    if "BoT_DR"    in sub else 0
            print(f"{mname:<12}{cond:<10}"
                  f"{c17:>8.3f}{c18:>8.3f}{uns:>8.3f}{bot:>8.3f}")

    # Diversity effect
    print("\nUNSW Diversity Effect (+Multi vs Single):")
    for mname in ["DAE","TrAE","Ensemble"]:
        key = mname.lower()
        df  = dfs.get(key)
        if df is None: continue
        s = df[df["condition"]=="Single"]["UNSW_AUC"].mean()
        m = df[df["condition"]=="Multi"]["UNSW_AUC"].mean()
        print(f"  {mname}: {s:.3f} -> {m:.3f} ({m-s:+.3f})")

    print("\nLODO Results (held-out F1, mean ± std):")
    for exp in ["LODO-1","LODO-2","LODO-3"]:
        sub = df_lodo[df_lodo["experiment"]==exp]
        ho  = sub["held_out"].iloc[0]
        print(f"  {exp} (held-out {ho}):")
        for mname in ["DAE","TrAE","Ensemble"]:
            ms = sub[sub["model"]==mname]["held_out_F1"]
            if len(ms): print(f"    {mname}: {ms.mean():.3f}±{ms.std(ddof=0):.3f}")


if __name__ == "__main__":
    log.info("Paper 3 — DAE + TransformerAE ensemble (300k size-matched)")
    X_s, X_m, V_s, V_m, tests, envs = load_data()
    log.info("Running main experiments...")
    dfs = run_main(X_s, X_m, V_s, V_m, tests)
    log.info("Running LODO experiments...")
    df_lodo = run_lodo(envs, tests)
    print_summary(dfs, df_lodo)
    log.info("All done.")
