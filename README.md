# A-GRCS Reproducible Code Release

Code and reproducibility artifacts for **Adaptive Graph-Restricted Risk Control for UAV Scheduling under Time-Varying Channel Uncertainty**.

## Repository contents

- `code/agrcs/` — core channel, calibration, scheduling, and learning modules.
- `code/run_experiments.py` — main experiment runner.
- `code/reproduce.py` — one-command quick or full reproduction workflow.
- `code/build_report.py` — generates the main publication figures only.
- `code/reviewer_experiments.py` — additional sensitivity and replication experiments.
- `code/reviewer_report.py` — generates separated supplementary figures and paired checks.
- `code/verify.py` — implementation and accounting checks.
- `code/test_figure_style.py` — figure-style validation.
- `models/` — retained trained model checkpoints and training logs.
- `results/` — retained numerical summaries and selected reproducibility outputs.
- `example_outputs/` — example figure and quick-run artifacts.
- `figures/` — generated PDF and PNG figures.

## Tested environment

- Windows 10/11 or Linux
- Python 3.13.5
- NumPy 2.3.5
- SciPy 1.17.0
- Matplotlib 3.10.8

## Installation

Create and activate a clean environment, then install the pinned dependencies:

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r code\requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
```

## Validation first

Run the implementation checks before experiments:

```bash
python code/verify.py
python code/test_figure_style.py
```

## Quick smoke reproduction

```bash
python code/reproduce.py --profile quick
```

This is intended to confirm that the environment and simulation pipeline run correctly. It does not reproduce every full experiment.

## Full reproduction

```bash
python code/reproduce.py --profile paper
```

The full workflow runs the main experiments, bulk-service experiment, latency benchmark, figure generation, review stress tests, additional sensitivity experiments, and system-model rendering.

To retrain the retained learning baseline before the full run:

```bash
python code/reproduce.py --profile paper --retrain
```

## Generate main figures from existing results

If complete reproduced results already exist:

```bash
python code/build_report.py --results results/reproduced
```

Generated figures are written to:

```text
figures/
```

Multi-panel results are exported as separate files, for example:

```text
feedback_a.pdf
feedback_b.pdf
shift_a.pdf
shift_b.pdf
stress_a.pdf
stress_b.pdf
scaling_a.pdf
scaling_b.pdf
energy_a.pdf
energy_b.pdf
```

## Generate supplementary figures

After the review experiments have been generated:

```bash
python code/reviewer_report.py --baseline-results results/reproduced
```

The supplementary figures are written to:

```text
figures/supplementary/
```

including separate panel files such as:

```text
risk_sensitivity_a.pdf
risk_sensitivity_b.pdf
review_sensitivity_existing_a.pdf
review_sensitivity_existing_b.pdf
review_sensitivity_existing_c.pdf
paired30_a.pdf
paired30_b.pdf
```

## Figure formatting

Generated plots use:

- serif/Times-compatible fonts;
- 9 pt axis labels;
- 8 pt tick labels and legends;
- complete top, bottom, left, and right borders;
- vector PDF output;
- 600 DPI PNG output.

## Reproducibility notes

The experiments use seeded synthetic channel simulations. Retained numerical summaries and model checkpoints are included to support validation and comparison. The quick profile is a smoke test; quantitative publication replication should use the full profile and the declared seed sets.
