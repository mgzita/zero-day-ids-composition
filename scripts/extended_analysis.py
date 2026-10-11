"""
extended_analysis.py
====================
Covers three remaining reviewer requirements:

B. Score correlation between models (complementarity)
C. PR-AUC, Precision, Recall, TPR@FPR=0.01, TPR@FPR=0.05
D. Paired Wilcoxon signed-rank tests (single vs multi)

Also covers:
A. Clean diversity experiment:
   Single: 200k CIC17
   Multi:  100k CIC17 + 100k CIC18
   Test:   UNSW (completely excluded)

Saves:
  results/score_correlation.csv
  results/extended_metrics.csv
  results/paired_stats.csv
  results/clean_diversity.csv

Expected time: ~6 hours
"""
import os, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve)
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEDUP    = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
OUT_DIR  = r"C:/MLProject/paper3_ensemble/results"
CKPT_DIR = r"C:/MLProject/paper3_ensemble/checkpoints"
SEEDS    = [42, 123, 456, 789, 999]
N        = 100_000
os.makedirs(OUT_DIR, exist_ok=True)


# ── Models ────────────────────────────────────────────────
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
        pos     = torch.arange(self.n_feat, device=x.device)
        tokens  = tokens + self.pos_emb(pos).unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.decoder(encoded).squeeze(-1)


class SVDDEncoder(nn.Module):
    def __init__(self, input_dim=18, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,64), nn.ReLU(),
            nn.Linear(64,32),        nn.ReLU(),
            nn.Linear(32,latent_dim, bias=False))
    def forward(self, x): return self.net(x)


def train_ae(model, Xtr, Xva, seed, noise=0.1):
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
            n = torch.clamp(b+torch.randn_like(b)*noise,-10,10)
            opt.zero_grad(); crit(model(n),b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            nv = torch.clamp(Xv+torch.randn_like(Xv)*noise,-10,10)
            vl = crit(model(nv),Xv).item()
        sch.step(vl)
        if vl < best:
            best=vl; bs={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=10: break
    model.load_state_dict(bs)
    return model


def get_ae_scores(model, X):
    model.eval(); out=[]
    with torch.no_grad():
        Xt=torch.FloatTensor(X)
        for i in range(0,len(Xt),10000):
            b=Xt[i:i+10000]
            out.append(((b-model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def load_svdd(seed, cond):
    """Load saved SVDD checkpoint."""
    path = os.path.join(CKPT_DIR, f"svdd_seed{seed}_{cond}.pt")
    if not os.path.exists(path):
        log.warning("No SVDD checkpoint: %s", path)
        return None, None
    ck = torch.load(path, map_location='cpu')
    enc = SVDDEncoder()
    enc.load_state_dict(ck['encoder_state'])
    enc.eval()
    return enc, ck['center']


def get_svdd_scores(enc, c, X):
    out=[]
    with torch.no_grad():
        Xt=torch.FloatTensor(X)
        for i in range(0,len(Xt),10000):
            b=Xt[i:i+10000]
            z=enc(b)
            dist=((z-c)**2).sum(dim=1)
            out.append(dist.numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip((s-p1)/(p99-p1+1e-8),0,1)


def extended_metrics(scores, yt, tau, name, seed, model, cond):
    """Compute full metric suite."""
    yp   = (scores > tau).astype(int)
    auc  = roc_auc_score(yt, scores)
    prauc= average_precision_score(yt, scores)
    f1   = f1_score(yt, yp, zero_division=0)
    prec = precision_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)
    fpr, tpr, _ = roc_curve(yt, scores)
    tpr_1  = float(np.interp(0.01, fpr, tpr))
    tpr_5  = float(np.interp(0.05, fpr, tpr))
    return {
        "seed": seed, "model": model,
        "condition": cond, "dataset": name,
        "ROC_AUC":   round(auc, 4),
        "PR_AUC":    round(prauc, 4),
        "F1":        round(f1, 4),
        "Precision": round(prec, 4),
        "Recall":    round(rec, 4),
        "TPR@FPR001":round(tpr_1, 4),
        "TPR@FPR005":round(tpr_5, 4),
    }


# ── Load data ──────────────────────────────────────────────
def load_all():
    log.info("Loading data...")
    d = np.load(os.path.join(DEDUP,"train_dedup.npz"))
    X_all = np.clip(d["X_train"],-10,10).astype(np.float32)
    n = len(X_all)//3
    X_c17=X_all[:n]; X_c18=X_all[n:2*n]; X_usw=X_all[2*n:]

    d = np.load(os.path.join(DEDUP,"val_dedup.npz"))
    V_all = np.clip(d["X_val"],-10,10).astype(np.float32)
    nv = len(V_all)//3
    V_c17=V_all[:nv]; V_c18=V_all[nv:2*nv]; V_usw=V_all[2*nv:]

    rng = np.random.default_rng(0)
    idx17=rng.choice(len(X_c17),N,replace=False)
    idx18=rng.choice(len(X_c18),N,replace=False)
    idxus=rng.choice(len(X_usw),N,replace=False)
    vi17 =rng.choice(len(V_c17),N//4,replace=False)
    vi18 =rng.choice(len(V_c18),N//4,replace=False)
    vius =rng.choice(len(V_usw),N//4,replace=False)

    # Multi-env (100k x 3)
    Xm = np.concatenate([X_c17[idx17],X_c18[idx18],X_usw[idxus]])
    Vm = np.concatenate([V_c17[vi17],V_c18[vi18],V_usw[vius]])

    # Single-env (300k CIC17)
    idx_s=rng.choice(len(X_c17),N*3,replace=False)
    vi_s =rng.choice(len(V_c17),N*3//4,replace=False)
    Xs = X_c17[idx_s]; Vs = V_c17[vi_s]

    # Clean diversity: single=200k CIC17, multi=100k CIC17+CIC18
    idx_cd_s=rng.choice(len(X_c17),N*2,replace=False)
    idx_cd_m17=rng.choice(len(X_c17),N,replace=False)
    idx_cd_m18=rng.choice(len(X_c18),N,replace=False)
    vi_cd_s =rng.choice(len(V_c17),N//2,replace=False)
    vi_cd_m =rng.choice(len(V_c17),N//4,replace=False)
    vi_cd_m2=rng.choice(len(V_c18),N//4,replace=False)
    X_cd_s = X_c17[idx_cd_s]
    V_cd_s = V_c17[vi_cd_s]
    X_cd_m = np.concatenate([X_c17[idx_cd_m17],X_c18[idx_cd_m18]])
    V_cd_m = np.concatenate([V_c17[vi_cd_m],V_c18[vi_cd_m2]])

    # Test sets
    tests={}
    for name,fname in [
        ("CIC17","test_cic17_dedup.npz"),
        ("CIC18","test_cic18_dedup.npz"),
        ("UNSW", "test_unsw_dedup.npz"),
    ]:
        d=np.load(os.path.join(DEDUP,fname))
        tests[name]=(np.clip(d["X"],-10,10).astype(np.float32),
                     d["y"].astype(int))
        log.info("  %s: %d rows", name, len(tests[name][0]))

    return (Xs,Vs,Xm,Vm,X_cd_s,V_cd_s,X_cd_m,V_cd_m,tests,
            X_c17,X_c18,X_usw,
            V_c17[vi17],V_c18[vi18],V_usw[vius])


# ── Main ───────────────────────────────────────────────────
def run():
    (Xs,Vs,Xm,Vm,X_cd_s,V_cd_s,X_cd_m,V_cd_m,tests,
     X_c17,X_c18,X_usw,Vc17,Vc18,Vusw) = load_all()

    ext_rows  = []  # extended metrics
    corr_rows = []  # score correlation
    stat_rows = []  # for paired tests
    cd_rows   = []  # clean diversity

    for seed in SEEDS:
        log.info("="*55)
        log.info("Seed %d", seed)
        t0 = time.time()

        # ── Train DAE and TrAE on multi-env ───────────────
        log.info("  DAE multi...")
        dae_m = train_ae(DAE(), Xm, Vm, seed)
        trs_d = get_ae_scores(dae_m, Xm[:50000])
        p1d,p99d = np.percentile(trs_d,1), np.percentile(trs_d,99)
        vs_d  = norm(get_ae_scores(dae_m,Vm),p1d,p99d)
        tau_d = float(np.percentile(vs_d,95))

        log.info("  TrAE multi...")
        trae_m = train_ae(TransformerAE(), Xm, Vm, seed)
        trs_t  = get_ae_scores(trae_m, Xm[:50000])
        p1t,p99t = np.percentile(trs_t,1), np.percentile(trs_t,99)
        vs_t   = norm(get_ae_scores(trae_m,Vm),p1t,p99t)
        tau_t  = float(np.percentile(vs_t,95))

        # Load SVDD multi checkpoint
        svdd_enc, svdd_c = load_svdd(seed, "Multi")

        # ── Extended metrics on test sets ──────────────────
        for ds,(Xt,yt) in tests.items():
            sd = norm(get_ae_scores(dae_m,Xt),p1d,p99d)
            st = norm(get_ae_scores(trae_m,Xt),p1t,p99t)
            se2= (sd+st)/2
            tau_e2 = float(np.percentile((vs_d+vs_t)/2, 95))

            ext_rows.append(extended_metrics(sd,yt,tau_d,  ds,seed,"DAE","Multi"))
            ext_rows.append(extended_metrics(st,yt,tau_t,  ds,seed,"TrAE","Multi"))
            ext_rows.append(extended_metrics(se2,yt,tau_e2,ds,seed,"Ens2","Multi"))

            if svdd_enc is not None:
                trs_sv = get_svdd_scores(svdd_enc,svdd_c,Xm[:50000])
                p1sv,p99sv = np.percentile(trs_sv,1),np.percentile(trs_sv,99)
                vs_sv = norm(get_svdd_scores(svdd_enc,svdd_c,Vm),p1sv,p99sv)
                tau_sv= float(np.percentile(vs_sv,95))
                ssv   = norm(get_svdd_scores(svdd_enc,svdd_c,Xt),p1sv,p99sv)
                se3   = (sd+st+ssv)/3
                tau_e3= float(np.percentile((vs_d+vs_t+vs_sv)/3,95))
                ext_rows.append(extended_metrics(ssv,yt,tau_sv,ds,seed,"SVDD","Multi"))
                ext_rows.append(extended_metrics(se3,yt,tau_e3,ds,seed,"Ens3","Multi"))

            # ── Score correlation ──────────────────────────
            pr_dt,_ = pearsonr(sd, st)
            corr_rows.append({
                "seed":seed, "dataset":ds,
                "pair":"DAE-TrAE",
                "pearson": round(float(pr_dt),4)})
            if svdd_enc is not None:
                pr_ds,_ = pearsonr(sd, ssv)
                pr_ts,_ = pearsonr(st, ssv)
                corr_rows.append({"seed":seed,"dataset":ds,
                    "pair":"DAE-SVDD","pearson":round(float(pr_ds),4)})
                corr_rows.append({"seed":seed,"dataset":ds,
                    "pair":"TrAE-SVDD","pearson":round(float(pr_ts),4)})

        # ── DAE single-env for paired tests ───────────────
        log.info("  DAE single...")
        dae_s = train_ae(DAE(), Xs, Vs, seed)
        trs_ds = get_ae_scores(dae_s, Xs[:50000])
        p1ds,p99ds = np.percentile(trs_ds,1),np.percentile(trs_ds,99)
        vs_ds  = norm(get_ae_scores(dae_s,Vs),p1ds,p99ds)
        tau_ds = float(np.percentile(vs_ds,95))
        Xt_u,yt_u = tests["UNSW"]
        auc_s = roc_auc_score(yt_u,
            norm(get_ae_scores(dae_s,Xt_u),p1ds,p99ds))
        auc_m_d = roc_auc_score(yt_u,
            norm(get_ae_scores(dae_m,Xt_u),p1d,p99d))
        stat_rows.append({"seed":seed,"model":"DAE",
            "single_UNSW_AUC":round(auc_s,4),
            "multi_UNSW_AUC": round(auc_m_d,4)})

        # ── Clean diversity experiment ─────────────────────
        log.info("  Clean diversity: DAE single 200k CIC17...")
        dae_cd_s = train_ae(DAE(), X_cd_s, V_cd_s, seed)
        trs_cds  = get_ae_scores(dae_cd_s, X_cd_s[:50000])
        p1_cds,p99_cds = np.percentile(trs_cds,1),np.percentile(trs_cds,99)
        auc_cd_s = roc_auc_score(yt_u,
            norm(get_ae_scores(dae_cd_s,Xt_u),p1_cds,p99_cds))

        log.info("  Clean diversity: DAE multi 100k CIC17+CIC18...")
        dae_cd_m = train_ae(DAE(), X_cd_m, V_cd_m, seed)
        trs_cdm  = get_ae_scores(dae_cd_m, X_cd_m[:50000])
        p1_cdm,p99_cdm = np.percentile(trs_cdm,1),np.percentile(trs_cdm,99)
        auc_cd_m = roc_auc_score(yt_u,
            norm(get_ae_scores(dae_cd_m,Xt_u),p1_cdm,p99_cdm))

        cd_rows.append({"seed":seed,
            "DAE_single_200k_CIC17": round(auc_cd_s,4),
            "DAE_multi_100k_CIC17_CIC18": round(auc_cd_m,4),
            "diff": round(auc_cd_m-auc_cd_s,4)})
        log.info("    Clean diversity: single=%.4f multi=%.4f diff=%+.4f",
                 auc_cd_s, auc_cd_m, auc_cd_m-auc_cd_s)

        log.info("Seed %d done %.1f min", seed, (time.time()-t0)/60)

    # ── Save results ───────────────────────────────────────
    df_ext  = pd.DataFrame(ext_rows)
    df_corr = pd.DataFrame(corr_rows)
    df_stat = pd.DataFrame(stat_rows)
    df_cd   = pd.DataFrame(cd_rows)

    df_ext.to_csv(os.path.join(OUT_DIR,"extended_metrics.csv"),  index=False)
    df_corr.to_csv(os.path.join(OUT_DIR,"score_correlation.csv"),index=False)
    df_stat.to_csv(os.path.join(OUT_DIR,"paired_stats.csv"),     index=False)
    df_cd.to_csv(os.path.join(OUT_DIR,"clean_diversity.csv"),    index=False)

    # ── Print summaries ────────────────────────────────────
    print("\n" + "="*65)
    print("EXTENDED METRICS (mean, 5 seeds, multi-env, UNSW)")
    print("="*65)
    usw = df_ext[df_ext["dataset"]=="UNSW"]
    for m in ["DAE","TrAE","SVDD","Ens2","Ens3"]:
        sub = usw[usw["model"]==m]
        if len(sub)==0: continue
        print(f"  {m:<8} AUC={sub['ROC_AUC'].mean():.3f} "
              f"PR={sub['PR_AUC'].mean():.3f} "
              f"F1={sub['F1'].mean():.3f} "
              f"Prec={sub['Precision'].mean():.3f} "
              f"Rec={sub['Recall'].mean():.3f} "
              f"TPR@1%={sub['TPR@FPR001'].mean():.3f}")

    print("\nSCORE CORRELATION (mean Pearson, multi-env, UNSW):")
    usw_c = df_corr[df_corr["dataset"]=="UNSW"]
    for pair in ["DAE-TrAE","DAE-SVDD","TrAE-SVDD"]:
        sub = usw_c[usw_c["pair"]==pair]
        if len(sub)>0:
            print(f"  {pair}: r={sub['pearson'].mean():.3f}")

    print("\nCLEAN DIVERSITY (UNSW, UNSW excluded from training):")
    print(f"  DAE single 200k CIC17:       "
          f"{df_cd['DAE_single_200k_CIC17'].mean():.3f}"
          f"±{df_cd['DAE_single_200k_CIC17'].std(ddof=0):.3f}")
    print(f"  DAE multi  100k CIC17+CIC18: "
          f"{df_cd['DAE_multi_100k_CIC17_CIC18'].mean():.3f}"
          f"±{df_cd['DAE_multi_100k_CIC17_CIC18'].std(ddof=0):.3f}")
    print(f"  Mean gain: {df_cd['diff'].mean():+.3f}")

    # Wilcoxon test
    from scipy.stats import wilcoxon
    s_vals = df_stat["single_UNSW_AUC"].values
    m_vals = df_stat["multi_UNSW_AUC"].values
    if len(s_vals) >= 5:
        try:
            stat, p = wilcoxon(m_vals, s_vals, alternative='greater')
            print(f"\nPaired Wilcoxon (single vs multi UNSW AUC):")
            print(f"  W={stat:.0f}  p={p:.4f}")
        except Exception as e:
            print(f"Wilcoxon: {e}")


if __name__ == "__main__":
    log.info("Extended analysis: B+C+D+A")
    run()
