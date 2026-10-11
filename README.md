# From Domain Shift to Zero-Day Resilience: Training-Environment Composition for Network Intrusion Detection

Code and results for:

> M. Zita, M. U. Rehman, M. Kazim, and A. H. Al-Bayatti. *From Domain Shift to Zero-Day Resilience: Training-Environment Composition for Network Intrusion Detection.* In Proc. International Conference on Electrical and Computer Engineering Researches (ICECER 2026), Istanbul, Türkiye, 3 to 5 December 2026.

The study tests whether combining architecturally different unsupervised models, trained under diverse conditions, improves detection in network environments the models have never seen. Three models are trained only on benign traffic under identical size-matched conditions: a Denoising Autoencoder (DAE), a Transformer Autoencoder (TrAE), and Deep SVDD. Two ensembles average their normalized scores: Ens-2 (DAE and TrAE) and Ens-3 (all three).

## Key results (five seeds)

| Finding | Result |
|---|---|
| UNSW-NB15 ROC-AUC, single-environment training | 0.444 to 0.570 |
| UNSW-NB15 ROC-AUC, multi-environment training that includes UNSW-NB15 benign traffic | 0.831 to 0.934 |
| Controlled test with UNSW-NB15 excluded from both conditions | no reliable improvement |
| Best model, CIC-IDS2017 held out (LODO-1) | Ens-3, AUC 0.857 |
| Best model, CSE-CIC-IDS2018 held out (LODO-2) | TrAE, AUC 0.944 |
| All models, UNSW-NB15 held out (LODO-3) | normalized AUC 0.44 to 0.50, F1 about 0.05 |
| DAE raw AUC under LODO-3 | 0.14 to 0.19; attacks score lower than benign traffic in all 20 seed and configuration combinations |

The gain on UNSW-NB15 comes from target-domain coverage rather than diversity alone, and neither ensemble consistently beats the best single model.

## Datasets

Download the datasets and place them under `data/` (not tracked):

- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

The scripts use the 18-feature harmonized space, IQR scaling, and deduplicated splits from [zero-day-ids-ensemble](https://github.com/mgzita/zero-day-ids-ensemble). Check the paths near the top of each script before running.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on Linux or macOS
pip install -r requirements.txt
```

## Reproducing the paper

Run scripts from the repository root. All results use seeds 42, 123, 456, 789, and 999.

### 1. Training

| Step | Script |
|---|---|
| Transformer Autoencoder | `scripts/train_transformer_ae.py` |
| DAE and TrAE, single and multi-environment | `scripts/train_paper3.py` |
| Deep SVDD | `scripts/train_svdd_paper3.py`, `scripts/run_missing_svdd.py` |

### 2. Leave-one-dataset-out (LODO)

| Step | Script |
|---|---|
| LODO-1, LODO-2, and LODO-3 | `scripts/run_lodo_paper3.py` |
| Individual folds and additional seeds | `scripts/run_lodo2_only.py`, `scripts/run_lodo3_only.py`, `scripts/run_lodo_789_999.py`, `scripts/rerun_seed999.py` |

### 3. Analysis

| Paper result | Script | Output in `results_fixed/` |
|---|---|---|
| Single versus multi-environment, LODO, extended metrics, clean diversity | `scripts/paper3_consolidated.py`, `scripts/targeted_raw_auc_fix.py` | `table3_single_multi.csv`, `table4_lodo.csv`, `table5_extended_metrics.csv`, `table7_clean_diversity.csv` |
| Extended metrics | `scripts/extended_analysis.py` | `table5_extended_metrics.csv` |
| Score correlation between models | `scripts/error_correlation.py` | `table6_score_correlation.csv` |
| Raw AUC and DAE score inversion | `scripts/svdd_raw_auc.py`, `scripts/raw_auc_summary_stats.py`, `scripts/diagnostic_all_seeds.py` | `svdd_raw_auc.csv`, `diagnostic_reversal_check.csv`, `table_lodo3_table7_raw_fixed.csv` |
| Rank-based fusion under LODO-3 | `scripts/lodo3_dump_scores.py`, `scripts/lodo3_rank_fusion.py` | `lodo3_rank_fusion.csv` |
| Single-environment raw AUC check | `scripts/check_table3_single_raw.py` | `table3_single_raw_check.csv` |

`scripts/smoke_test.py` runs a quick end-to-end check of the pipeline.

## License

MIT License

## Contact

Margaret Zita, De Montfort University, Leicester, United Kingdom
