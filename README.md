# Multimodal Edge AI for Firefighter Safety - Simulation Code

Supporting data and code for the manuscript:

> *Multimodal Edge Artificial Intelligence and Physiological Sensing for Firefighter Safety in Hazardous Environments*

## Authors

- Arpan Bom
- Sushanta Khadka

## Contents

| Path | Description |
|------|-------------|
| `simulate_and_evaluate.py` | End-to-end synthetic mission simulation, model fit, baselines, metrics, and figure generation |
| `requirements.txt` | Python dependencies |
| `results/metrics.json` | Numeric evaluation metrics (seed 42) |
| `results/summary.md` | Human-readable results summary |
| `figures/` | Figures 1-4 used in the manuscript |
| `manuscript/JOURNAL_MANUSCRIPT.md` | Journal manuscript draft |

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python simulate_and_evaluate.py
```

Fixed random seed: **42**.

## Important disclaimer

All results in this repository are from a **controlled synthetic simulation** (12 firefighter sorties x 60 minutes).
**No real fireground recordings, department logs, or human physiological datasets were used.**

## Citation

If you use this code or metrics, please cite the accompanying manuscript and this repository.

## License

MIT License (see `LICENSE`).
