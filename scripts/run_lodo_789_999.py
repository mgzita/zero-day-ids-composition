"""
run_lodo_paper3.py
===================
Paper 3: LODO-1, LODO-2, LODO-3 experiments.
Train on CIC17+CIC18, evaluate on UNSW-NB15.
DAE, TrAE, and Ensemble.
3 seeds: 42, 123, 456.

Run in a SEPARATE PowerShell window while
train_paper3.py is still running.

Output:
  paper3_ensemble/results/lodo_789_999_results.csv

Expected time: ~30 hours (3 configs x 5 seeds x ~120 min)
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

SEEDS = [789, 999]
NOISE = 0.1
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
    def __init__(self, n_feat=18, d_model=64,
                 n_heads=4, d_ff=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb    = nn.Embedding(n_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True)
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


def train_model(model, Xtr, Xva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    crit = nn.HuberLoss(delta=1.0)
    Xt   = torch.FloatTensor(Xtr)
    Xv   = torch.FloatTensor(Xva)
    best, pat, bs = float('inf'), 0, None
    for ep in range(100):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = Xt[idx[i:i+512]]
            n = torch.clamp(b + torch.randn_like(b)*NOISE, -10, 10)
            opt.zero_grad(); crit(model(n), b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            nv = torch.clamp(Xv+torch.randn_like(Xv)*NOISE,-10,10)
            vl = crit(model(nv), Xv).item()
        sch.step(vl)
        if vl < best:
            best = vl
            bs = {k:v.clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 10:
                log.info("    Early stop ep %d", ep+1); break
    model.load_state_dict(bs)
    return model


def get_scores(model, X):
    model.eval(); out=[]
    with torch.no_grad():
        Xt = torch.FloatTensor(X)
        for i in range(0, len(Xt), 10000):
            b = Xt[i:i+10000]
            out.append(((b-model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip((s-p1)/(p99-p1+1e-8), 0, 1)


def run():
    log.info("LODO 1-3: DAE + TrAE + Ensemble")
    log.info("Seeds: %s", SEEDS)

    # Load all training environments
    d = np.load(os.path.join(DEDUP, "train_dedup.npz"))
    X_all = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n = len(X_all) // 3
    X_c17 = X_all[:n]; X_c18 = X_all[n:2*n]; X_usw = X_all[2*n:]

    d = np.load(os.path.join(DEDUP, "val_dedup.npz"))
    V_all = np.clip(d["X_val"], -10, 10).astype(np.float32)
    nv = len(V_all) // 3
    V_c17 = V_all[:nv]; V_c18 = V_all[nv:2*nv]; V_usw = V_all[2*nv:]

    # Load test sets
    tests = {}
    for name, fname in [
        ("CIC17","test_cic17_dedup.npz"),
        ("CIC18","test_cic18_dedup.npz"),
        ("UNSW", "test_unsw_dedup.npz"),
    ]:
        d = np.load(os.path.join(DEDUP, fname))
        tests[name] = (np.clip(d["X"],-10,10).astype(np.float32),
                       d["y"].astype(int))
        log.info("Loaded %s: %d rows", name, len(tests[name][0]))

    # Subsample 100k per env
    rng = np.random.default_rng(0)
    idx = {e: rng.choice(len(X), N_PER_ENV, replace=False)
           for e, X in [("c17",X_c17),("c18",X_c18),("usw",X_usw)]}
    vidx = {e: rng.choice(len(V), N_PER_ENV//4, replace=False)
            for e, V in [("c17",V_c17),("c18",V_c18),("usw",V_usw)]}

    envs = {
        "CIC17": (X_c17[idx["c17"]], V_c17[vidx["c17"]]),
        "CIC18": (X_c18[idx["c18"]], V_c18[vidx["c18"]]),
        "UNSW":  (X_usw[idx["usw"]], V_usw[vidx["usw"]]),
    }

    lodo_configs = [
        ("LODO-1", ["CIC18","UNSW"], "CIC17"),
        ("LODO-2", ["CIC17","UNSW"], "CIC18"),
        ("LODO-3", ["CIC17","CIC18"],"UNSW"),
    ]

    rows = []

    for exp, train_envs, held_out in lodo_configs:
        log.info("="*55)
        log.info("%s: train=%s  held-out=%s",
                 exp, "+".join(train_envs), held_out)

        Xtr = np.concatenate([envs[e][0] for e in train_envs])
        Xva = np.concatenate([envs[e][1] for e in train_envs])
        X_test, y_test = tests[held_out]
        log.info("Train: %d  Val: %d  Test: %d",
                 len(Xtr), len(Xva), len(X_test))

        for seed in SEEDS:
            log.info("  Seed %d...", seed); t0 = time.time()

            # DAE
            dae = train_model(DAE(), Xtr, Xva, seed)
            trs = get_scores(dae, Xtr[:50000])
            p1d=float(np.percentile(trs,1)); p99d=float(np.percentile(trs,99))
            vs_d = norm(get_scores(dae, Xva), p1d, p99d)
            tau_d = float(np.percentile(vs_d, 95))
            s_dae = norm(get_scores(dae, X_test), p1d, p99d)
            auc_d = roc_auc_score(y_test, s_dae)
            f1_d  = f1_score(y_test, (s_dae>tau_d).astype(int), zero_division=0)
            log.info("    DAE:  AUC=%.4f F1=%.4f", auc_d, f1_d)

            # TrAE
            trae = train_model(TransformerAE(), Xtr, Xva, seed)
            trs = get_scores(trae, Xtr[:50000])
            p1t=float(np.percentile(trs,1)); p99t=float(np.percentile(trs,99))
            vs_t = norm(get_scores(trae, Xva), p1t, p99t)
            tau_t = float(np.percentile(vs_t, 95))
            s_trae = norm(get_scores(trae, X_test), p1t, p99t)
            auc_t = roc_auc_score(y_test, s_trae)
            f1_t  = f1_score(y_test, (s_trae>tau_t).astype(int), zero_division=0)
            log.info("    TrAE: AUC=%.4f F1=%.4f", auc_t, f1_t)

            # Ensemble
            s_ens = (s_dae + s_trae) / 2
            tau_e = (tau_d + tau_t) / 2
            auc_e = roc_auc_score(y_test, s_ens)
            f1_e  = f1_score(y_test, (s_ens>tau_e).astype(int), zero_division=0)
            log.info("    Ens:  AUC=%.4f F1=%.4f", auc_e, f1_e)

            log.info("  Seed %d done %.1f min", seed, (time.time()-t0)/60)

            for mname, auc, f1 in [
                ("DAE",auc_d,f1_d),("TrAE",auc_t,f1_t),("Ensemble",auc_e,f1_e)
            ]:
                rows.append({"experiment":exp,"held_out":held_out,
                             "model":mname,"seed":seed,
                             "AUC":auc,"F1":f1})

    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "lodo_789_999_results.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)

    print("\n" + "="*65)
    print("LODO RESULTS (mean ± pop std, 5 seeds)")
    print("="*65)
    print(f"{'Experiment':<12}{'Model':<12}{'AUC':>12}{'F1':>10}")
    print("-"*48)
    for exp,_,ho in lodo_configs:
        for mname in ["DAE","TrAE","Ensemble"]:
            sub = df[(df["experiment"]==exp)&(df["model"]==mname)]
            auc = sub["AUC"].values; f1 = sub["F1"].values
            print(f"{exp:<12}{mname:<12}"
                  f"{auc.mean():.3f}±{auc.std(ddof=0):.3f}"
                  f"{f1.mean():>8.3f}±{f1.std(ddof=0):.3f}")
        print()


if __name__ == "__main__":
    run()
