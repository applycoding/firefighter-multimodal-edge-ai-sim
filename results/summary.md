# SYNTHETIC Validation Study: Firefighter Helmet Multimodal Edge-AI

## Disclaimer

All data in this study are SYNTHETIC. No real fireground recordings, department logs, or human physiological datasets were used. Results illustrate methodological behavior under controlled induced events only and must not be interpreted as operational performance claims.

## Design

- N = 12 simulated firefighter sorties (crew-members)
- T = 60 mission minutes at 1 sample / 10 s (360 timesteps per unit)
- Random seed = 42
- Latent hazards: external_heat_risk, toxicity_risk, physiological_strain, fall_or_mayday_risk, localization_uncertainty
- Induced events: heat ramp/flashover-precursor (4 units), CO spike (3), physio overload (3), fall (2), GPS-denied indoor DR drift (6), ambient noise/radio confound (4; should not alone trigger alerts)
- Ground-truth alert when latent primary risk >= 0.35; proposed alerts when P(risk > tau) >= 0.8

## Methods

1. Baseline A: Independent channel absolute thresholds (OR-union)
2. Baseline B: Univariate EWMA union rule (lambda=0.2, z>2.2)
3. Proposed: Multimodal Kalman filter + RTS smoother with empirical-Bayes hierarchical shrinkage of unit prior means; env_noise excluded from fusion
4. Ablations: thermal+gas only; physio-only; IMU(+GPS unc)-only

## Key Metrics

| Method | F1 | AUROC | Precision | Recall | Mean Delay (min) | FAR |
|--------|----|-------|-----------|--------|------------------|-----|
| BaselineA_Thresholds | 0.885 | 0.905 | 0.972 | 0.813 | 0.29 | 0.004 |
| BaselineB_EWMA | 0.903 | 0.993 | 0.936 | 0.873 | 1.21 | 0.010 |
| Proposed_Kalman | 0.842 | 0.997 | 1.000 | 0.727 | 1.24 | 0.000 |
| Ablation_ThermalGas | 0.635 | 0.856 | 1.000 | 0.465 | 2.31 | 0.000 |
| Ablation_Physio | 0.503 | 0.846 | 1.000 | 0.336 | 5.83 | 0.000 |
| Ablation_IMU | 0.147 | 0.662 | 0.538 | 0.085 | 5.71 | 0.011 |

## Honest Outcome Check

- Proposed beats best baseline on F1: False (Proposed F1=0.842 vs A=0.885, B=0.903)
- Proposed beats best baseline on AUROC: True (Proposed AUROC=0.997 vs A=0.905, B=0.993)
- Proposed has lower or equal FAR vs best baseline: True (Proposed FAR=0.000)
- Risk estimate vs true primary: Pearson r=0.965, RMSE=0.083
- FAR during noise-confound / non-event periods: 0.000 (env_noise not used in proposed fusion)

## Event-Specific Delays (Proposed)

- heat_ramp: n=4, mean delay=0.10 min, miss_rate=0.0
- co_spike: n=3, mean delay=0.08 min, miss_rate=0.0
- physio_overload: n=3, mean delay=0.08 min, miss_rate=0.0
- fall: n=2, mean delay=0.07 min, miss_rate=0.0

## Artifacts

- fig1: /workspace/firefighter-edge-ai-sim/figures/fig1_latent_vs_estimates.png
- fig2: /workspace/firefighter-edge-ai-sim/figures/fig2_roc_comparison.png
- fig3: /workspace/firefighter-edge-ai-sim/figures/fig3_delay_falsealarms.png
- fig4: /workspace/firefighter-edge-ai-sim/figures/fig4_modality_coverage.png
- metrics_json: /workspace/firefighter-edge-ai-sim/results/metrics.json

## Reproducibility

Run: `python simulate_and_evaluate.py` from this directory with packages in requirements.txt. Prefer Kalman/RTS + empirical Bayes (no MCMC).
