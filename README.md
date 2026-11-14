# PACT: Propensity-weighted Audit-fed Conformal Triage

Code, experiment outputs and manuscript sources for

> Sneha, Prabh Deep Singh, Narinder Kaur. *Conformal triage with selective laboratory verification for drifting biosensor and bioprocess streams
> Prepared for Frontiers in Bioengineering and
> Biotechnology, Research Topic "Trustworthy and Secure AI Frameworks for Bioengineering: Explainability,
> Risk Assessment, and Biosafety".

PACT turns a classifier for drifting biosensor / bioprocess streams into a triage system. Each sample gets a
conformal prediction set. Unambiguous non-hazard singletons are released automatically, everything else is
escalated to laboratory or expert verification, and a small random share of released samples is audited.
Verified outcomes are fed back into calibration with inverse-propensity weights, so that the threshold reflects
the whole stream and not only the samples that were chosen for verification.

## Repository layout

```
src/
  preprocessing.py      raw data -> initial pool + ordered deployment stream (one .npz per dataset)
  conformal.py          nonconformity scores, weighted conformal quantile
  engine.py             prequential streaming engine; every method is a configuration of it
  proposed_model.py     PACT configuration and ablations
  baseline_models.py    Split CP, APS, ACI, NexCP, conformal P-control, uncertainty sampling, random audit
  risk_assessment.py    triage rule and decision record for a single sample
  uncertainty.py        thin public API for scores / quantiles / coverage-debt controller
  explainability.py     exact TreeSHAP, deletion faithfulness, stability
  robustness.py         sensor noise, sensor dropout, bounded evasion (defensive stress tests)
  evaluation.py         all metrics
  results_io.py         reader for the append-only metric files
experiments/
  run_all.py            main / ablation / sensitivity / robust / poison grids (resumable)
  static_classifiers.py conventional and deep-ensemble baselines
  explain.py            explanation-quality experiment
  make_tables.py        LaTeX tables + number macros from results/metrics
  make_figures.py       result figures;  make_supp_figures.py, fig_diagrams.py
  configurations/       grid definition used for the paper
results/metrics/        raw per-run metrics (CSV), one row per dataset x method x seed
results/logs/           per-sample logs of the main comparison (coverage, set size, audit, prediction)
figures/                all figures used in the manuscript and supplement
datasets/               download instructions; no raw data is redistributed
```

## Reproducing the results

```bash
pip install -r requirements.txt
# 1. download the five datasets (see datasets/README.md) and preprocess
python -m src.preprocessing <raw_root> datasets/processed
# 2. experiments (each command is resumable; rows already present are skipped)
python -m experiments.run_all main        gas,home,wine,ferment,water 1,2,3,4,5
python -m experiments.run_all ablation    gas,home,ferment,water      1,2,3
python -m experiments.run_all robust      home 1,2,3 && python -m experiments.run_all robust gas 1,2
python -m experiments.run_all poison      home 1,2,3 && python -m experiments.run_all poison gas 1,2
python -m experiments.run_all sensitivity gas 1,2
python -m experiments.static_classifiers  gas,home,wine,ferment,water 1,2,3,4,5
python -m experiments.explain             gas,ferment,water 1,2,3
# 3. tables, figures, manuscript
python -m experiments.make_tables && python -m experiments.make_figures
python -m experiments.make_supp_figures && python -m experiments.fig_diagrams
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The full grid takes roughly three hours on two CPU cores; the gas-drift and fermentation streams dominate.
Seeds 1-5 are used for reported results. Seed 0 was used during development and is excluded.

## Minimal use

```python
from src.engine import load, run
from src.proposed_model import PACT
from src.evaluation import summarise

log = run(load("gas"), PACT, seed=1)
print(summarise(log))          # coverage, set size, label rate, accuracy of automatic decisions, ...
```

## Notes on scope

* Latency values are measured computation times on the test machine. No network delay is simulated.
* The water-reuse hazard label (intact cell concentration >= 1e5 cells/mL) is an operational proxy defined
  for the study, not a regulatory limit.
* The stress tests perturb sensor features only (noise, missing channel, bounded shift, corrupted
  verification labels). The repository contains no biological sequence data or protocols.
* Triage outputs are decision support. They are not a validated regulatory or clinical determination.

## Licence

Code: MIT (see `LICENSE`). Datasets remain under the terms of their original providers.
