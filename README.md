# CounterSignal

Performance monitoring for deployed clinical risk models that corrects for the interventions their alerts trigger, with uncertainty intervals, small sample verdicts and sequential drift alarms.

## The problem

A readmission or deterioration model flags a patient, the care team intervenes, and the event the model predicted does not happen. Standard monitoring compares predictions with recorded outcomes, so a model that is working looks miscalibrated, and the more effective the intervention, the larger the apparent error. Monitoring only the untreated patients helps, but they are mostly low risk, so a change concentrated in the patients who get treated is easy to miss. Small hospitals face a second problem: too few cases for monthly metrics to mean anything.

CounterSignal evaluates the model against the outcome it was validated to predict, the outcome under standard care, and says plainly when the data cannot answer the question.

## What it reports each month

* **Observed/expected events, AUC, calibration intercept and slope, Brier score** for three views: all patients (standard), untreated patients only, and untreated patients weighted by the inverse probability of remaining untreated (CounterSignal). A doubly robust observed/expected is reported alongside.
* **A verdict with an interval**: *stable*, *drift* or *not enough evidence yet*, from where the 95% bootstrap interval sits against an acceptable range. For the last case, an estimate of how many effectively untreated patients would settle it.
* **A cumulative alarm** (two sided CUSUM) whose monthly inputs are standardized against their exact simulated null distribution, so the false alarm rate holds in small windows. The limit is set for a 5% chance of any false alarm over the planned horizon.
* **Early signals while outcomes are pending**: population stability index of the score and each covariate against the validation data.
* **Diagnostics**: effective sample size of the weights and the number of capped weights, so positivity problems are visible.
* **A plain language report card** for clinical leads.

## Benchmark on real hospital records

69,973 first diabetic inpatient encounters from the UCI Diabetes 130-US Hospitals dataset (1999 to 2008). A logistic regression readmission model (validation AUC 0.631) is deployed in simulation: patients in the top 10% of risk are alerted, clinicians treat 70% of them, and treatment prevents 40% of their readmissions. The recorded outcomes stand in for what would have happened without treatment, so the true performance in every month is known. 200 simulated deployments of 12 months per scenario; all three monitors use the same alarm rule and limit.

| | Standard | Untreated only | CounterSignal |
|---|---|---|---|
| False alarm, model unchanged (planned 5%) | 96.5% | 10.0% | 5.5% |
| False alarm, sicker case mix, model unchanged | 100.0% | 12.5% | 1.5% |
| False alarm, small hospital (150 patients a month) | 11.5% | 6.5% | 3.0% |
| Real drift among alerted patients: caught | 57.0% | 65.0% | 97.5% |
| Real drift: false alarm before it began | 26.5% | 0.0% | 0.0% |
| Real drift: median month of first alarm | 5 | 5 | 3 |

The correction has a cost: reweighting leaves about 1,217 effective patients out of 2,000 a month, monthly intervals are wider, and single months usually return "not enough evidence yet". The cumulative alarm carries the evidence across months. The full design, sensitivity runs (20% and 60% treatment effect) and limits are in [docs/benchmark.md](docs/benchmark.md).

The treatment process is simulated and every driver of treatment is logged, which is the best case for the method's main assumption. See [docs/method.md](docs/method.md) for the estimators and every assumption.

## Quick start

```bash
pip install git+https://github.com/Boatengs/countersignal
```

```python
from countersignal import Monitor, MonitorConfig
from countersignal.report import to_markdown

config = MonitorConfig(
    score="score",            # model's predicted risk
    outcome="outcome",        # 0/1, blank until recorded
    treated="treated",        # 0/1, intervention given
    window="month",
    covariates=["alerted", "prior_admissions", "age"],  # what drove treatment
    oe_band=(0.8, 1.25),
)
result = Monitor(reference_df, config).run(deployment_log_df)
print(result.table[["window", "oe_ipw", "oe_ipw_lo", "oe_ipw_hi", "oe_verdict", "cusum_alarm"]])
print(to_markdown(result))
```

`reference_df` is the model's validation data from before deployment (score, outcome and the covariates). `deployment_log_df` has one row per scored patient after deployment. `result.table` has one row per period, labelled in its `window` column whatever the input column is called.

## Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard/Home.py
```

Pages: benchmark results, a month by month explorer of simulated deployments, a page to run CounterSignal on your own CSV files (or a synthetic demo), and the method notes.

## Repository layout

| Path | Contents |
|---|---|
| `src/countersignal/` | monitor, weighted metrics, propensity weights, sequential alarm, shift signals, report card |
| `src/countersignal/datasets/uci_diabetes.py` | dataset loader and cohort rules |
| `benchmarks/run_benchmark.py` | semi-synthetic benchmark |
| `results/` | benchmark outputs committed by the workflow |
| `dashboard/` | Streamlit app |
| `docs/` | method, benchmark case study, release notes |
| `tests/` | known truth tests for estimators, alarm calibration and the loader |

## Reproducibility

The *Data and benchmark* workflow downloads the archive from UCI, records its SHA-256 in `data/manifest.json` (and fails if it ever changes), runs the benchmark and commits the results. *Quality checks* runs Ruff and the test suite on every push.

## Data and citation

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J. Licensed CC BY 4.0; a copy of the archive is kept in `data/raw/`.

Method references: Feng et al., *Monitoring machine learning (ML) based risk prediction algorithms in the presence of confounding medical interventions* (arXiv:2211.09781); Keogh and van Geloven, *Prediction under interventions: evaluation of counterfactual performance using longitudinal observational data* (Epidemiology, 2024).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
