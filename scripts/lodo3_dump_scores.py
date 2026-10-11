"""
lodo3_dump_scores.py
=====================
Saves PER-FLOW raw (pre-normalization) scores for LODO-3
(train = CIC17 + CIC18, held out = UNSW) for ONE seed, so the
benign-vs-attack score-distribution figure can be drawn.

Put this file in the same folder as paper3_consolidated.py and run:
    python lodo3_dump_scores.py

It REUSES paper3_consolidated.py (same data subsampling, same fixed
training function, same seeded init), so seed 42 should reproduce the
paper's LODO-3 numbers. A check at the end compares the raw DAE AUC with
the value already in table_lodo3_table7_raw_fixed.csv (seed 42: 0.1522).

Time: about 1 to 1.5 hours on CPU (trains one DAE and one TrAE).
Set TRAIN_TRAE = False to skip the TrAE and finish faster.

Output: OUT_DIR/lodo3_scores_seed{SEED}.npz  (upload this file)
"""
import os, logging
import numpy as np
from sklearn.metrics import roc_auc_score

import paper3_consolidated as p3

SEED       = 42
TRAIN_TRAE = True

log = logging.getLogger(__name__)


def main():
    (X_single, V_single, X_multi, V_multi, tests, envs,
     X_cd_s, V_cd_s, X_cd_m, V_cd_m) = p3.load_all()

    # LODO-3 exactly as in paper3_consolidated.run()
    Xtr = np.concatenate([envs["CIC17"][0], envs["CIC18"][0]])
    Xva = np.concatenate([envs["CIC17"][1], envs["CIC18"][1]])
    X_test, y_test = tests["UNSW"]
    log.info("LODO-3: train=%d val=%d test=%d (benign %d, attack %d)",
             len(Xtr), len(Xva), len(y_test),
             (y_test == 0).sum(), (y_test == 1).sum())

    out = {"y": y_test.astype(np.int8), "seed": np.array(SEED)}

    # ---- DAE ----
    dae, ep = p3.train_ae(p3.DAE, Xtr, Xva, SEED)
    log.info("DAE stopped at epoch %d", ep)
    tr_ref = p3.sample_rows(Xtr, 50000, SEED)
    out["dae_train_raw"] = p3.get_ae_scores(dae, tr_ref).astype(np.float32)
    out["dae_raw"]       = p3.get_ae_scores(dae, X_test).astype(np.float32)
    out["dae_p1"]  = np.percentile(out["dae_train_raw"], 1)
    out["dae_p99"] = np.percentile(out["dae_train_raw"], 99)
    auc = roc_auc_score(y_test, out["dae_raw"])
    log.info("DAE raw AUC = %.4f  (paper CSV, seed 42: 0.1522)", auc)

    # ---- TrAE (contrast panel) ----
    if TRAIN_TRAE:
        trae, ep_t = p3.train_ae(p3.TransformerAE, Xtr, Xva, SEED)
        log.info("TrAE stopped at epoch %d", ep_t)
        out["trae_train_raw"] = p3.get_ae_scores(trae, tr_ref).astype(np.float32)
        out["trae_raw"]       = p3.get_ae_scores(trae, X_test).astype(np.float32)
        out["trae_p1"]  = np.percentile(out["trae_train_raw"], 1)
        out["trae_p99"] = np.percentile(out["trae_train_raw"], 99)
        log.info("TrAE raw AUC = %.4f  (paper CSV, seed 42: 0.6852)",
                 roc_auc_score(y_test, out["trae_raw"]))

    # ---- SVDD (existing checkpoint, not retrained) ----
    enc, c = p3.load_svdd(SEED, "LODO-3", "UNSW")
    if enc is not None:
        out["svdd_train_raw"] = p3.get_svdd_scores(enc, c, tr_ref).astype(np.float32)
        out["svdd_raw"]       = p3.get_svdd_scores(enc, c, X_test).astype(np.float32)
        out["svdd_p1"]  = np.percentile(out["svdd_train_raw"], 1)
        out["svdd_p99"] = np.percentile(out["svdd_train_raw"], 99)
        log.info("SVDD raw AUC = %.4f  (paper CSV, seed 42: 0.6033)",
                 roc_auc_score(y_test, out["svdd_raw"]))

    b, a = out["dae_raw"][y_test == 0], out["dae_raw"][y_test == 1]
    log.info("DAE raw mean error: benign %.4f  attack %.4f", b.mean(), a.mean())

    path = os.path.join(p3.OUT_DIR, f"lodo3_scores_seed{SEED}.npz")
    np.savez_compressed(path, **out)
    log.info("Saved %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    main()
