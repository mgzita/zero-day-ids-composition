# Architecture-by-Target Mechanisms of Training-Environment Diversity in Unsupervised Anomaly Detection

Code for:

> M. Zita, M. U. Rehman, M. Abrar, M. Kazim, and S. Khalid. *Architecture-by-Target Mechanisms of Training-Environment Diversity in Unsupervised Anomaly Detection.* Manuscript in preparation, De Montfort University, 2026.

This study follows [zero-day-ids-ensemble](https://github.com/mgzita/zero-day-ids-ensemble), which showed that training on benign traffic from several environments improves detection in an unseen environment. Here we ask *how* that diversity changes model behavior. Three kinds of evidence are compared: feature attribution (KernelSHAP and Integrated Gradients), functional feature ablation, and representation-level distance (MMD² and sliced Wasserstein).

## Design

| Element | Setting |
|---|---|
| Models | Autoencoder (AE), Denoising Autoencoder (DAE), Deep SVDD, Normalizing Flow |
| Target domains (never seen in training) | UNSW-NB15, IEC-104 (industrial control traffic), SMD (server telemetry, native 38-feature space) |
| Training conditions | Single (one source) and Diverse (two sources, size-matched to Single) |
| Network training sources | CIC-IDS2017 and CSE-CIC-IDS2018 |
| Seeds | Ten per model and target combination |
| Statistics | Paired t-test, Wilcoxon signed-rank, paired Cohen's d, Benjamini–Hochberg false discovery rate correction across 48 tests |

## Key results

- 13 of 48 tests survive false discovery rate correction.
- KernelSHAP showed apparent effects for four of twelve model and target combinations; none was reproduced with Integrated Gradients.
- Functional ablation effects depend on both model and target. Deep SVDD reverses from Sensitive-feature dominance on UNSW-NB15 (d = 3.04) to Stable-feature dominance on IEC-104 (d = −1.08); the Normalizing Flow shows the same reversal between UNSW-NB15 and SMD.
- The apparent IEC-104 Normalizing Flow reversal in MMD² is a radial basis function kernel bandwidth artifact. Sliced Wasserstein distance shows a narrower representation gap under Diverse training.
- The model by target interaction is significant (Wald F(6, 79) = 57.427, p < 0.00001).

## Datasets

Place raw files under `raw_data/` (not tracked):

- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- IEC-104: see the dataset reference in the paper
- SMD (Server Machine Dataset): https://github.com/NetManAIOps/OmniAnomaly

Paths and frozen design settings are in `preprocessing/config.py`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on Linux or macOS
pip install -r requirements.txt
```

## Reproducing the paper

Run all commands from the repository root.

### 1. Preprocessing

| Step | Script |
|---|---|
| Dataset loaders | `preprocessing/loaders/*.py` |
| IEC-104 and SMD harmonization | `preprocessing/harmonize_iec104.py`, `preprocessing/harmonize_smd.py` |
| Final preprocessing and audit | `preprocessing/finalize_preprocessing.py`, `preprocessing/audit_cleaned_data.py` |

### 2. Training and full pipelines

| Target | AE, DAE, Deep SVDD | Normalizing Flow |
|---|---|---|
| UNSW-NB15 | `train_ae_dae_lodo3_checkpoints.py`, `extend_ae_dae_svdd_unsw_h1_repr.py` | `train_normalizing_flow.py`, `flow_full_pipeline.py` |
| IEC-104 | `ae_dae_svdd_pipeline_iec104.py` | `flow_full_pipeline_iec104.py` |
| SMD | `ae_dae_svdd_pipeline_smd.py` | `flow_full_pipeline_smd.py` |

### 3. Analyses

| Paper result | Script |
|---|---|
| Integrated Gradients attribution | `compute_ig_attribution_all.py`, `analyze_ig_results.py` |
| Attribution tables (SHAP and IG) | `generate_h1_latex_tables.py` |
| Functional ablation | `ablation_h3_lodo3.py`, `ablation_h3_lodo3_extra_seeds.py`, `combine_h3_lodo3_seeds.py` |
| Decorrelated feature groups | `ablation_h4_v2_decorrelated.py` |
| Representation distance (MMD²) | `compute_flow_representation_stats.py` |
| Representation distance (sliced Wasserstein) | `compute_flow_representation_stats_wasserstein.py`, `representation_interaction_wasserstein.py` |
| MMD² bandwidth artifact diagnosis | `flow_iec104_mmd_bandwidth_sweep.py`, `flow_iec104_repr_diagnostic.py` |
| SMD statistics | `analyze_smd_results.py`, `analyze_flow_smd_results.py` |
| Noise dose-response sweep | `noise_dose_response_sweep.py`, `noise_trend_test.py` |
| Fallback scorer usage | `fallback_usage_tally.py` |
| Combine all results | `final_combine_all_results.py` |
| Check every number in the manuscript | `verify_all_manuscript_numbers.py` |

## License

MIT License

## Contact

Margaret Zita, De Montfort University, Leicester, United Kingdom
