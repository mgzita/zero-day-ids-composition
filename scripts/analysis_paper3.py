"""
analysis_paper3.py
==================
Combines three post-training analyses for Paper 3:

1. Correct ensemble threshold (Point 4)
   Derive tau_ens from ensemble validation scores
   not average of individual thresholds

2. Extended metrics (Point 6)
   PR-AUC, Precision, Recall, TPR@FPR=0.01
   Paired t-test and 95% CI across seeds

Requires:
  paper3_ensemble/results/main_results.csv
  paper3_ensemble/results/lodo_paper3_results.csv
  dedup npz files

Output:
  paper3_ensemble/results/ensemble_corrected.csv
  paper3_ensemble/results/extended_metrics.csv
  paper3_ensemble/results/paired_stats_*.csv

Note: Distribution shift (Point 5) uses
Paper 2 MMD values directly:
  CIC17 vs CIC18: MMD=0.029
  CIC17 vs UNSW:  MMD=0.102
  CIC18 vs UNSW:  MMD=0.088

Expected time: ~3 hours
"""
import os, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, f1_score,
    average_precision_score, precision_score,
    recall_score)
from scipy import stats

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP   = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
OUT_DIR = r"C:/MLProject/paper3_ensemble/results"
SEEDS   = [42, 123, 456, 789, 999]
N_PER_ENV = 100_000


# ── Models (needed to recompute scores) ───────────────────
class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(18,64),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,16),nn.ReLU(),
            nn.Linear(16,8), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(8,16), nn.ReLU(),
            nn.Linear(16,32),nn.ReLU(),
            nn.Linear(32,64),nn.ReLU(),
            nn.Linear(64,18))
    def forward(self,x): return self.dec(self.enc(x))


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
            nn.Linear(d_model,d_ff),nn.ReLU(),
            nn.Linear(d_ff,1))
        self.n_feat = n_feat

    def forward(self, x):
        tokens  = self.input_proj(x.unsqueeze(-1))
        pos     = torch.arange(self.n_feat,
                               device=x.device)
        tokens  = tokens + self.pos_emb(pos).unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.decoder(encoded).squeeze(-1)


def train_model(model, Xtr, Xva, seed, noise=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    opt  = torch.optim.Adam(model.parameters(),lr=1e-3)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(
               opt, patience=5)
    crit = nn.HuberLoss(delta=1.0)
    Xt   = torch.FloatTensor(Xtr)
    Xv   = torch.FloatTensor(Xva)
    best, pat, bs = float('inf'), 0, None
    for ep in range(100):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0,len(Xt),512):
            b=Xt[idx[i:i+512]]
            n=torch.clamp(b+torch.randn_like(b)*noise,-10,10)
            opt.zero_grad(); crit(model(n),b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            nv=torch.clamp(Xv+torch.randn_like(Xv)*noise,-10,10)
            vl=crit(model(nv),Xv).item()
        sch.step(vl)
        if vl<best:
            best=vl
            bs={k:v.clone() for k,v in model.state_dict().items()}
            pat=0
        else:
            pat+=1
            if pat>=10: break
    model.load_state_dict(bs)
    return model


def get_scores(model, X):
    model.eval(); out=[]
    with torch.no_grad():
        Xt=torch.FloatTensor(X)
        for i in range(0,len(Xt),10000):
            b=Xt[i:i+10000]
            out.append(((b-model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def norm(s,p1,p99):
    return np.clip((s-p1)/(p99-p1+1e-8),0,1)


# ── 1. Corrected ensemble threshold ───────────────────────
def compute_corrected_ensemble(Xtr, Xva, tests, seed):
    """
    Train DAE and TrAE, compute ensemble scores,
    then derive threshold from ENSEMBLE validation
    scores (not average of individual thresholds).
    """
    dae  = train_model(DAE(), Xtr, Xva, seed)
    trae = train_model(TransformerAE(), Xtr, Xva, seed)

    # DAE normalization
    trs_d = get_scores(dae, Xtr[:50000])
    p1d=float(np.percentile(trs_d,1))
    p99d=float(np.percentile(trs_d,99))

    # TrAE normalization
    trs_t = get_scores(trae, Xtr[:50000])
    p1t=float(np.percentile(trs_t,1))
    p99t=float(np.percentile(trs_t,99))

    # Ensemble validation scores
    vs_d = norm(get_scores(dae,  Xva), p1d, p99d)
    vs_t = norm(get_scores(trae, Xva), p1t, p99t)
    vs_ens = (vs_d + vs_t) / 2

    # CORRECTED: threshold from ensemble val scores
    tau_ens_corrected = float(np.percentile(vs_ens, 95))

    # OLD method: average of individual thresholds
    tau_d   = float(np.percentile(vs_d, 95))
    tau_t   = float(np.percentile(vs_t, 95))
    tau_ens_old = (tau_d + tau_t) / 2

    results = {}
    for name, (Xt, yt) in tests.items():
        if name == "BoT": continue
        sd = norm(get_scores(dae,  Xt), p1d, p99d)
        st = norm(get_scores(trae, Xt), p1t, p99t)
        se = (sd + st) / 2

        auc = roc_auc_score(yt, se)

        # Old threshold
        f1_old = f1_score(yt, (se>tau_ens_old).astype(int),
                          zero_division=0)
        # Corrected threshold
        f1_new = f1_score(yt, (se>tau_ens_corrected).astype(int),
                          zero_division=0)

        results[name] = {
            'AUC': auc,
            'F1_old_tau': f1_old,
            'F1_corrected_tau': f1_new,
            'tau_old': tau_ens_old,
            'tau_corrected': tau_ens_corrected,
        }
        log.info("    %s: AUC=%.4f F1_old=%.4f "
                 "F1_corrected=%.4f",
                 name, auc, f1_old, f1_new)

    return results


# ── 2. Extended metrics ────────────────────────────────────
def extended_metrics(scores, yt, tau, name):
    """Compute full metric suite."""
    yp  = (scores > tau).astype(int)
    auc = roc_auc_score(yt, scores)
    prauc = average_precision_score(yt, scores)
    f1  = f1_score(yt, yp, zero_division=0)
    prec = precision_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)

    # TPR at FPR=0.01
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(yt, scores)
    tpr_at_fpr01 = float(np.interp(0.01, fpr, tpr))

    return {
        "dataset": name,
        "ROC_AUC": round(auc, 4),
        "PR_AUC":  round(prauc, 4),
        "F1":      round(f1, 4),
        "Precision": round(prec, 4),
        "Recall":  round(rec, 4),
        "TPR@FPR001": round(tpr_at_fpr01, 4),
    }


def compute_paired_stats(df, metric="ROC_AUC"):
    """Paired t-test and 95% CI for single vs multi."""
    results = []
    for model in ["DAE","TrAE","Ensemble"]:
        s_vals = df[(df["model"]==model) &
                    (df["condition"]=="Single")][metric].values
        m_vals = df[(df["model"]==model) &
                    (df["condition"]=="Multi")][metric].values
        if len(s_vals) < 2: continue
        diff   = m_vals - s_vals
        t_stat, p_val = stats.ttest_rel(m_vals, s_vals)
        ci_lo  = diff.mean() - 2.776 * diff.std(ddof=1)/np.sqrt(5)
        ci_hi  = diff.mean() + 2.776 * diff.std(ddof=1)/np.sqrt(5)
        results.append({
            "model": model,
            "mean_diff": round(diff.mean(), 4),
            "CI_95_lo":  round(ci_lo, 4),
            "CI_95_hi":  round(ci_hi, 4),
            "t_stat":    round(t_stat, 3),
            "p_value":   round(p_val, 4),
        })
    return pd.DataFrame(results)


# ── Main ───────────────────────────────────────────────────
def run():
    log.info("Loading data...")
    d     = np.load(os.path.join(DEDUP,"train_dedup.npz"))
    X_all = np.clip(d["X_train"],-10,10).astype(np.float32)
    n     = len(X_all)//3
    X_c17 = X_all[:n]
    X_c18 = X_all[n:2*n]
    X_usw = X_all[2*n:]

    d     = np.load(os.path.join(DEDUP,"val_dedup.npz"))
    V_all = np.clip(d["X_val"],-10,10).astype(np.float32)
    nv    = len(V_all)//3

    rng   = np.random.default_rng(0)
    idx17 = rng.choice(n, N_PER_ENV, replace=False)
    idx18 = rng.choice(n, N_PER_ENV, replace=False)
    idxus = rng.choice(n, N_PER_ENV, replace=False)
    vi17  = rng.choice(nv, N_PER_ENV//4, replace=False)
    vi18  = rng.choice(nv, N_PER_ENV//4, replace=False)
    vius  = rng.choice(nv, N_PER_ENV//4, replace=False)

    Xtr = np.concatenate([X_c17[idx17],
                          X_c18[idx18],
                          X_usw[idxus]])
    Xva = np.concatenate([V_all[:nv][vi17],
                          V_all[nv:2*nv][vi18],
                          V_all[2*nv:][vius]])

    tests = {}
    for name, fname in [
        ("CIC17","test_cic17_dedup.npz"),
        ("CIC18","test_cic18_dedup.npz"),
        ("UNSW", "test_unsw_dedup.npz"),
    ]:
        d = np.load(os.path.join(DEDUP,fname))
        tests[name] = (
            np.clip(d["X"],-10,10).astype(np.float32),
            d["y"].astype(int))

    # ── Analysis 1: Corrected ensemble threshold ──
    log.info("="*55)
    log.info("1. Corrected ensemble threshold")
    ens_rows = []
    for seed in SEEDS:
        log.info("  Seed %d...", seed)
        res = compute_corrected_ensemble(
            Xtr, Xva, tests, seed)
        for ds, vals in res.items():
            ens_rows.append({
                "seed": seed,
                "dataset": ds,
                **vals})

    df_ens = pd.DataFrame(ens_rows)
    df_ens.to_csv(os.path.join(
        OUT_DIR,"ensemble_corrected.csv"),
        index=False)
    log.info("Saved ensemble_corrected.csv")

    print("\nCORRECTED vs OLD THRESHOLD (mean 5 seeds):")
    for ds in ["CIC17","CIC18","UNSW"]:
        sub = df_ens[df_ens["dataset"]==ds]
        print(f"  {ds}: F1_old={sub['F1_old_tau'].mean():.3f} "
              f"F1_corrected={sub['F1_corrected_tau'].mean():.3f} "
              f"AUC={sub['AUC'].mean():.3f}")

    # ── Analysis 2: Paired stats from main results ─
    # Note: Distribution shift uses Paper 2 values:
    #   CIC17 vs CIC18: MMD=0.029
    #   CIC17 vs UNSW:  MMD=0.102 (largest)
    #   CIC18 vs UNSW:  MMD=0.088
    log.info("MMD distances from Paper 2:")
    log.info("  CIC17-CIC18=0.029, CIC17-UNSW=0.102,"
             " CIC18-UNSW=0.088")
    log.info("  UNSW is most distant -> explains LODO-3")

    # ── Analysis 2 continued: Paired stats ─
    log.info("="*55)
    log.info("3. Paired statistics")
    main_csv = os.path.join(OUT_DIR,"main_results.csv")
    if os.path.exists(main_csv):
        df_main = pd.read_csv(main_csv)
        for ds in ["CIC17_AUC","CIC18_AUC","UNSW_AUC"]:
            if ds in df_main.columns:
                df_main["metric"] = ds
                df_main["ROC_AUC"] = df_main[ds]
                stats_df = compute_paired_stats(df_main)
                print(f"\nPaired stats ({ds}):")
                print(stats_df.to_string(index=False))
                stats_df.to_csv(os.path.join(
                    OUT_DIR,
                    f"paired_stats_{ds}.csv"),
                    index=False)

    log.info("All analyses complete.")


if __name__ == "__main__":
    run()
