# Multimodal Edge Artificial Intelligence and Physiological Sensing for Firefighter Safety in Hazardous Environments

**Authors:** Arpan Bom; Sushanta Khadka

**Article type:** Original Research (Methods + Simulation Study)

**Suggested venues:** *Sensors*; *IEEE Sensors Journal*; *Fire Technology*; *International Journal of Environmental Research and Public Health*

---

## Abstract

**Background.** Wildland and structural firefighting expose crews to rapidly changing thermal, toxic, and physiological hazards. Network connectivity is often unavailable exactly when alerts matter most.

**Methods.** We present a helmet-oriented multimodal sensing architecture that fuses thermal, gas, light, GPS/IMU, and physiological channels with edge AI for local hazard detection and a non-critical cloud mission-control layer for incident command. Safety-critical inference is designed to operate offline. We evaluate a linear-Gaussian multimodal Kalman filter with RTS smoothing and empirical-Bayes unit shrinkage in a controlled **synthetic** study (12 sorties x 60 minutes; seed 42) against independent thresholds, univariate EWMA, and modality ablations. **No real fireground or department data were used.**

**Results.** The proposed tracker achieved AUROC = 0.997 and false-alert rate = 0.000, recovering latent primary risk with Pearson r = 0.965 (RMSE = 0.083). At a conservative P(risk > tau) >= 0.8 operating point, F1 = 0.842 (precision = 1.000, recall = 0.727), trailing EWMA on F1 (0.903) while eliminating false alerts. Thermal+gas, physio-only, and IMU-only ablations underperformed the full multimodal model (AUROC 0.856 / 0.846 / 0.662). Ambient noise confound alone did not trigger proposed alerts.

**Conclusions.** Multimodal edge state-space fusion can improve ranking quality and suppress false alerts under simulated missingness and GPS denial, supporting an architecture that keeps safety-critical functions onboard. Translation requires analog and field evaluation before operational claims.

**Keywords:** firefighter safety; edge AI; multimodal sensing; Kalman filter; physiological monitoring; PPE; FirstNet

---

## 1. Introduction

Firefighters operating in active wildfire or structural fire conditions manage more information than any person can process cleanly under stress: shifting smoke, wind, heat, crew location, and their own exhaustion. Traditional protective ensembles protect the body (National Fire Protection Association [NFPA], 2018) but do little to surface early hazard cues. Sudden cardiac events remain a leading cause of line-of-duty death in the U.S. fire service (Smith, Barr, & Kales, 2013), and heat strain under thermal protective clothing is a documented operational risk (Hostler et al., 2010; Kim, Coca, Williams, & Roberge, 2011). Wearable sensing in firefighter PPE is therefore an active area, with hard constraints on comfort, durability, and signal quality (Shakeriaski & Ghodrat, 2022; Coca et al., 2010).

This paper describes a helmet-mounted multimodal system that combines thermal imaging proxies, gas and light sensing, GPS with IMU dead reckoning, and physiological monitoring with edge AI, while keeping audio as the primary interaction channel. Prior multimodal firefighter support systems show that fusing inertial, environmental, and toxicity cues can support fall and hazard alerting (Pham et al., 2019; Chai et al., 2021). Our contributions are: (i) an architecture that separates offline safety-critical edge inference from optional cloud mission control; (ii) a formal multimodal state-space alert model; and (iii) an honest synthetic validation with baselines, ablations, and operating characteristics. We do not claim fireground-certified performance.

---

## 2. Related Work

Firefighter physiological and rehab literature motivates continuous monitoring under PPE (Smith et al., 2013; Hostler et al., 2010; Kim et al., 2011; Bustos et al., 2021; Taborri et al., 2021). Sensor-fusion systems for on-duty firefighters and fall detection illustrate multimodal edge pipelines (Pham et al., 2019; Chai et al., 2021). Indoor location for first responders remains a public-safety research priority (National Institute of Standards and Technology [NIST], 2021). Nationwide public-safety broadband provides a backdrop for optional command uplinks where coverage exists (First Responder Network Authority, 2024). Embedded deep learning for firefighting assistance further motivates on-device inference under link loss (Bhattarai et al., 2020). PPE and fire-brigade standards frame deployment constraints (NFPA, 2018; Occupational Safety and Health Administration [OSHA], n.d.).

---

## 3. Methods

### 3.1 System architecture (summary)

The helmet integrates thermal, light, gas, environmental, physiological, IMU, and GPS channels. Safety-critical fusion and alerting run onboard. A separate 5G/4G mission-control layer provides multi-unit maps and assignment sync without becoming a dependency for local alerts. Voice I/O is primary for hands-free interaction. Companion apps support configuration and after-action review. Standards and training programs remain complementary (NFPA, 2018; OSHA, n.d.; International Association of Fire Fighters [IAFF], n.d.).

### 3.2 Multimodal latent-risk model

For unit k and time t, let latent primary risk x_{k,t} evolve as a Gaussian random walk / VAR(1) state. Multimodal observations y_{k,t} (thermal, ambient temperature, CO, light, HR, skin temperature, IMU magnitude, GPS uncertainty) are linear-Gaussian emissions with missingness. Environmental noise is recorded for confound analysis but excluded from fusion. Unit-specific prior means are shrunk toward a population mean (empirical Bayes).

### 3.3 Inference and alerting

We use Kalman filtering and RTS smoothing. Alerts fire when P(x_{k,t} > tau | data) >= 0.8, with tau calibrated on early windows. This is approximate Bayesian inference (Kalman + empirical Bayes), not full MCMC.

### 3.4 Baselines and ablations

1. Independent absolute channel thresholds (OR-union)
2. Univariate EWMA union rule
3. Ablations: thermal+gas; physio-only; IMU(+GPS unc)-only

### 3.5 Synthetic evaluation design

N = 12 sorties, T = 60 minutes, 1 sample / 10 s, seed 42. Induced events: heat/flashover-precursor (4), CO spike (3), physio overload (3), fall (2), GPS-denied DR drift (6), ambient noise confound (4). Ground-truth elevated risk when latent primary risk >= 0.35. Metrics: F1, AUROC, precision, recall, mean detection delay, false-alert rate, Pearson r / RMSE vs true risk.

**Ethics:** No human subjects data and no department operational records were used.

---

## 4. Results

### 4.1 Overall performance

Table 1 summarizes detection metrics. The proposed multimodal Kalman tracker achieved **AUROC = 0.997** and **FAR = 0.000**, versus AUROC 0.905 / 0.993 and FAR 0.004 / 0.010 for thresholds and EWMA. At the conservative P>=0.8 operating point, Proposed **F1 = 0.842** trailed EWMA F1 = 0.903 because of lower recall (0.727 vs 0.873) with perfect precision (1.000). This is an intentional precision-first tradeoff for alert fatigue.

**Table 1.** Synthetic detection performance (12 x 60 min; seed 42).

| Method | F1 | AUROC | Precision | Recall | Mean delay (min) | FAR |
|--------|-----|-------|-----------|--------|------------------|-----|
| Independent thresholds | 0.885 | 0.905 | 0.972 | 0.813 | 0.29 | 0.004 |
| Univariate EWMA | 0.903 | 0.993 | 0.936 | 0.873 | 1.21 | 0.010 |
| **Proposed multimodal Kalman** | **0.842** | **0.997** | **1.000** | **0.727** | **1.24** | **0.000** |
| Ablation: thermal+gas | 0.635 | 0.856 | 1.000 | 0.465 | 2.31 | 0.000 |
| Ablation: physio-only | 0.503 | 0.846 | 1.000 | 0.336 | 5.83 | 0.000 |
| Ablation: IMU-only | 0.147 | 0.662 | 0.538 | 0.085 | 5.71 | 0.011 |

### 4.2 Latent tracking and events

Estimated primary risk matched synthetic truth with Pearson **r = 0.965** and **RMSE = 0.083** (Figure 1). Event-onset delays for heat, CO, physio, and fall were near-immediate once hazards rose (mean about 0.07-0.10 min for proposed onset detection). Mean day-level delay across elevated periods was 1.24 min.

### 4.3 Ablations and confounds

Full multimodality outperformed thermal+gas, physio-only, and IMU-only ablations on AUROC and F1 (Figure 2-3). High ambient noise / radio chatter alone did not produce proposed alerts (noise-confound FAR = 0.000) because env_noise is excluded from fusion. Figure 4 summarizes modality coverage under missingness.

---

## 5. Discussion

These synthetic results support keeping multimodal fusion at the edge: ranking quality and false-alert control improve when thermal, gas, physiological, and inertial cues share a latent risk state, consistent with prior firefighter multi-sensor systems (Pham et al., 2019; Chai et al., 2021). Separating offline safety-critical inference from optional FirstNet/5G mission control matches operational reality where connectivity fails under smoke and remote terrain (First Responder Network Authority, 2024).

**Honest limits.** (1) Synthetic dynamics only. (2) Approximate Kalman inference, not full hierarchical MCMC. (3) Oracle latent labels unavailable in the field. (4) Single seed, modest N. (5) No human-in-the-loop voice-alert usability study. (6) No NFPA/OSHA certification evaluation. Until analog and field trials exist, this manuscript should be read as methods + simulation, not an operational clearance study.

---

## 6. Conclusion

We presented a multimodal edge-AI architecture for firefighter helmet sensing and evaluated a shared-latent Kalman risk tracker in a controlled synthetic mission. Relative to channel-wise rules, the tracker improved AUROC and eliminated false alerts at a conservative operating point, while modality ablations confirmed the value of fusion. Field validation remains the required next step.

---

## Data and Code Availability

Simulation code, metrics, and figures are publicly available at:

**https://github.com/applycoding/firefighter-multimodal-edge-ai-sim** Re-run with seed 42. No real fireground data were used.

## Author Contributions

Arpan Bom and Sushanta Khadka contributed to conceptualization, methodology, simulation study design, manuscript drafting, and revision.

## Funding

[To be completed.]

## Acknowledgments

[To be completed.]

## Conflicts of Interest

The authors declare no competing interests.

---

## References

Bhattarai, M., Jensen-Curtis, A. R., & Martinez-Ramon, M. (2020). An embedded deep learning system for augmented reality in firefighting applications. In *Proceedings of the 19th IEEE International Conference on Machine Learning and Applications (ICMLA)* (pp. 1224-1230). IEEE. https://doi.org/10.1109/ICMLA51294.2020.00193

Bustos, D., Guedes, J. C., Baptista, J. S., Vaz, M., Costa, J. T., & Fernandes, R. J. (2021). Applicability of physiological monitoring systems within occupational groups: A systematic review. *Sensors, 21*(21), 7249. https://doi.org/10.3390/s21217249

Chai, X., Wu, R., Pike, M., Jin, H., Chung, W.-Y., & Lee, B.-G. (2021). Smart wearables with sensor fusion for fall detection in firefighting. *Sensors, 21*(20), 6770. https://doi.org/10.3390/s21206770

Coca, A., Roberge, R. J., Williams, W. J., Landsittel, D. P., Powell, J. B., & Palmiero, A. (2010). Physiological monitoring in firefighter ensembles: Wearable plethysmographic sensor vest versus standard equipment. *Journal of Occupational and Environmental Hygiene, 7*(2), 109-114. https://doi.org/10.1080/15459620903455722

First Responder Network Authority. (2024). *Fiscal Year 2023 annual report to Congress*. U.S. Department of Commerce, National Telecommunications and Information Administration. https://www.firstnet.gov/sites/default/files/FirstNetAuthority_AnnualReport_FY2023.pdf

Hostler, D., Reis, S. E., Bednez, J. C., Kerin, S., & Suyama, J. (2010). Comparison of active cooling devices with passive cooling for rehabilitation of firefighters performing exercise in thermal protective clothing: A report from the Fireground Rehab Evaluation (FIRE) trial. *Prehospital Emergency Care, 14*(3), 300-309. https://doi.org/10.3109/10903121003770654

International Association of Fire Fighters. (n.d.). *Fire Ground Survival (FGS) & fire fighter rescue training*. https://www.iaff.org/fire-ground-survival/

Kim, J.-H., Coca, A., Williams, W. J., & Roberge, R. J. (2011). Effects of liquid cooling garments on recovery and performance time in individuals performing strenuous work wearing a firefighter ensemble. *Journal of Occupational and Environmental Hygiene, 8*(7), 409-416. https://doi.org/10.1080/15459624.2011.584840

National Fire Protection Association. (2018). *NFPA 1971: Standard on protective ensembles for structural fire fighting and proximity fire fighting* (2018 ed.). NFPA.

National Institute of Standards and Technology. (2021, April 20). *PSCR awards $8M for First Responder 3D Indoor Tracking Prize*. NIST. https://www.nist.gov/news-events/news/2021/04/pscr-awards-8m-first-responder-3d-indoor-tracking-prize

Occupational Safety and Health Administration. (n.d.). *Fire brigades* (29 C.F.R. § 1910.156). U.S. Department of Labor. https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.156

Pham, V. T., Le, Q. B., Nguyen, D. A., Dang, N. D., Huynh, H. T., & Tran, D.-T. (2019). Multi-sensor data fusion in a real-time support system for on-duty firefighters. *Sensors, 19*(21), 4746. https://doi.org/10.3390/s19214746

Shakeriaski, F., & Ghodrat, M. (2022). Challenges and limitation of wearable sensors used in firefighters' protective clothing. *Journal of Fire Sciences, 40*(3), 214-245. https://doi.org/10.1177/07349041221079004

Smith, D. L., Barr, D. A., & Kales, S. N. (2013). Extreme sacrifice: Sudden cardiac death in the US Fire Service. *Extreme Physiology & Medicine, 2*, Article 6. https://doi.org/10.1186/2046-7648-2-6

Taborri, J., Pasinetti, S., Cardinali, L., Perroni, F., & Rossi, S. (2021). Preventing and monitoring work-related diseases in firefighters: A literature review on sensor-based systems and future perspectives in robotic devices. *International Journal of Environmental Research and Public Health, 18*(18), 9723. https://doi.org/10.3390/ijerph18189723
