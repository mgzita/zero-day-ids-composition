"""
rerun_seed999.py
================
Reruns seed 999 multi-environment only
for DAE, TrAE, and Ensemble.
Saves to paper3_ensemble/results/seed999_results.csv

Expected time: ~3 hours
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

DEDUP   = r"C:/MLProject/zero_day_project/artefacts_v2/dedup"
OUT_DIR = r"C:/MLProject/paper3_ensemble/results"
os.makedirs(OUT_DIR, exist_ok=True)

SEED      = 999
NOISE     = 0.1
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


def train_model(model, Xtr, Xva):
    torch.manual_seed(SEED); np.random.seed(SEED)
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
            n = torch.clamp(b+torch.randn_like(b)*NOISE,-10,10)
            opt.zero_grad(); crit(model(n),b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            nv = torch.clamp(Xv+torch.randn_like(Xv)*NOISE,-10,10)
            vl = crit(model(nv),Xv).item()
        sch.step(vl)
        if vl < best:
            best=vl; bs={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=10: log.info("  Early stop ep %d",ep+1); break
    model.load_state_dict(bs); return model


def get_scores(model, X):
    model.eval(); out=[]
    with torch.no_grad():
        Xt=torch.FloatTensor(X)
        for i in range(0,len(Xt),10000):
            b=Xt[i:i+10000]
            out.append(((b-model(b))**2).mean(dim=1).numpy())
    return np.concatenate(out)


def norm(s, p1, p99):
    return np.clip((s-p1)/(p99-p1+1e-8),0,1)


def run():
    log.info("Rerunning seed 999 multi-environment only")

    # Load data
    d = np.load(os.path.join(DEDUP,"train_dedup.npz"))
    X_all = np.clip(d["X_train"],-10,10).astype(np.float32)
    n = len(X_all)//3

    d = np.load(os.path.join(DEDUP,"val_dedup.npz"))
    V_all = np.clip(d["X_val"],-10,10).astype(np.float32)
    nv = len(V_all)//3

    # Subsample 100k per env
    rng = np.random.default_rng(0)
    idx17 = rng.choice(n, N_PER_ENV, replace=False)
    idx18 = rng.choice(n, N_PER_ENV, replace=False)
    idxus = rng.choice(n, N_PER_ENV, replace=False)
    vi17  = rng.choice(nv, N_PER_ENV//4, replace=False)
    vi18  = rng.choice(nv, N_PER_ENV//4, replace=False)
    vius  = rng.choice(nv, N_PER_ENV//4, replace=False)

    Xtr = np.concatenate([X_all[:n][idx17],
                          X_all[n:2*n][idx18],
                          X_all[2*n:][idxus]])
    Xva = np.concatenate([V_all[:nv][vi17],
                          V_all[nv:2*nv][vi18],
                          V_all[2*nv:][vius]])
    log.info("Multi train: %d  val: %d", len(Xtr), len(Xva))

    # Load test sets
    tests = {}
    for name, fname in [
        ("CIC17","test_cic17_dedup.npz"),
        ("CIC18","test_cic18_dedup.npz"),
        ("UNSW", "test_unsw_dedup.npz"),
        ("BoT",  "test_botiot_dedup.npz"),
    ]:
        d = np.load(os.path.join(DEDUP, fname))
        tests[name] = (np.clip(d["X"],-10,10).astype(np.float32),
                       d["y"].astype(int))

    rows = []

    # Train DAE
    log.info("Training DAE seed 999..."); t0=time.time()
    dae = train_model(DAE(), Xtr, Xva)
    trs = get_scores(dae, Xtr[:50000])
    p1d=float(np.percentile(trs,1)); p99d=float(np.percentile(trs,99))
    vs_d = norm(get_scores(dae,Xva),p1d,p99d)
    tau_d = float(np.percentile(vs_d,95))
    dae_scores = {}
    for name,(Xt,yt) in tests.items():
        s = norm(get_scores(dae,Xt),p1d,p99d)
        dae_scores[name] = s
        if name=="BoT":
            v = float((s>tau_d).mean())
            log.info("  DAE %s DR=%.4f", name, v)
            rows.append({"model":"DAE","seed":999,
                         f"{name}_AUC":float('nan'),f"{name}_DR":v})
        else:
            auc = roc_auc_score(yt,s)
            log.info("  DAE %s AUC=%.4f", name, auc)
    log.info("DAE done %.1f min",(time.time()-t0)/60)

    # Train TrAE
    log.info("Training TrAE seed 999..."); t0=time.time()
    trae = train_model(TransformerAE(), Xtr, Xva)
    trs = get_scores(trae, Xtr[:50000])
    p1t=float(np.percentile(trs,1)); p99t=float(np.percentile(trs,99))
    vs_t = norm(get_scores(trae,Xva),p1t,p99t)
    tau_t = float(np.percentile(vs_t,95))
    trae_scores = {}
    for name,(Xt,yt) in tests.items():
        s = norm(get_scores(trae,Xt),p1t,p99t)
        trae_scores[name] = s
        if name=="BoT":
            v = float((s>tau_t).mean())
            log.info("  TrAE %s DR=%.4f", name, v)
        else:
            auc = roc_auc_score(yt,s)
            log.info("  TrAE %s AUC=%.4f", name, auc)
    log.info("TrAE done %.1f min",(time.time()-t0)/60)

    # Ensemble
    tau_e = (tau_d+tau_t)/2
    print("\nSEED 999 MULTI-ENV RESULTS:")
    print("="*50)
    for name,(Xt,yt) in tests.items():
        s_ens = (dae_scores[name]+trae_scores[name])/2
        s_dae = dae_scores[name]
        s_trae = trae_scores[name]
        if name=="BoT":
            print(f"  BoT DR: DAE={float((s_dae>tau_d).mean()):.4f} "
                  f"TrAE={float((s_trae>tau_t).mean()):.4f} "
                  f"Ens={float((s_ens>tau_e).mean()):.4f}")
        else:
            print(f"  {name}: DAE={roc_auc_score(yt,s_dae):.4f} "
                  f"TrAE={roc_auc_score(yt,s_trae):.4f} "
                  f"Ens={roc_auc_score(yt,s_ens):.4f}")

    # Save
    import json
    result = {"seed": 999, "condition": "Multi"}
    for name,(Xt,yt) in tests.items():
        s_d = dae_scores[name]; s_t = trae_scores[name]
        s_e = (s_d+s_t)/2
        if name=="BoT":
            result[f"DAE_{name}_DR"]  = float((s_d>tau_d).mean())
            result[f"TrAE_{name}_DR"] = float((s_t>tau_t).mean())
            result[f"Ens_{name}_DR"]  = float((s_e>tau_e).mean())
        else:
            result[f"DAE_{name}_AUC"]  = roc_auc_score(yt,s_d)
            result[f"TrAE_{name}_AUC"] = roc_auc_score(yt,s_t)
            result[f"Ens_{name}_AUC"]  = roc_auc_score(yt,s_e)

    out = os.path.join(OUT_DIR,"seed999_results.json")
    with open(out,"w") as f: json.dump(result,f,indent=2)
    log.info("Saved -> %s", out)


if __name__=="__main__":
    run()
