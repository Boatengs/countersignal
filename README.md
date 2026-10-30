# CounterSignal

Performance monitoring for deployed clinical risk models that corrects for the interventions their alerts trigger, with uncertainty intervals, small sample verdicts and sequential drift alarms.

## The problem

A readmission or deterioration model flags a patient, the care team intervenes, and the event the model predicted does not happen. Standard monitoring compares predictions with recorded outcomes, so a model that is working looks miscalibrated, and the more effective the intervention, the larger the apparent error. Monitoring only the untreated patients helps, but they are mostly low risk, so a change concentrated in the patients who get treated is easy to miss. Small hospitals face a second problem: too few cases for monthly metrics to mean anything.

CounterSignal evaluates the model against the outcome it was validated to predict, the outcome under standard care, and says plainly when the data cannot answer the question.

## What it reports each month

* **Observed/expected events and AUC, with intervals, for three views**: all patients (standard), untreated patients only, and untreated patients weighted by the inverse probability of remaining untreated (CounterSignal). For the CounterSignal view also the calibration intercept and slope, the Brier score and a doubly robust observed/expected.
* **A verdict with an interval**: *stable*, *drift* or *not enough evidence yet*, from where the 95% bootstrap interval sits against an acceptable range. For the last case, an estimate of how large a month would need to be for the interval to reach the edge of the range.
* **A cumulative alarm** (two sided CUSUM) whose monthly inputs are standardized against a simulated null distribution (2,000 Monte Carlo draws given the month's scores and weights, with no normal approximation), which keeps it valid in small windows. The limit is set for a 5% chance of any false alarm over the planned horizon (12 months by default).
* **Early signals while outcomes are pending**: population stability index of the score, and of the covariate that has shifted most, against the validation data, with binary and discrete variables compared level by level.
* **Diagnostics**: effective sample size of the weights, the number of capped weights, and a "no untreated patients" verdict for a month in which every patient was treated; the capped weights count flags weaker positivity problems.
* **A plain language report card** for clinical leads.

## Benchmark on real patient records

Two public datasets, each with a logistic regression and a gradient boosting model: 30 day readmission for 69,973 diabetic inpatients from 130 U.S. hospitals (UCI, 1999 to 2008) and in hospital death for 110,204 sepsis admissions in Norway (UCI, 2011 to 2012). Deployment is simulated on top of the real outcomes: patients at or above the validation 90th percentile of risk (about 10 to 11% of patients) are alerted; clinicians treat about four in five alerted and one in twenty other patients, more often for patients with more prior inpatient stays (diabetes) or older patients (sepsis); and treatment prevents 40% of the events it is applied to. Each month is drawn from a known population, so the model's true performance in every month is known. 200 simulated deployments of 12 months per scenario. Four monitors are compared: standard (all recorded outcomes), untreated patients only, an independent implementation of the Feng et al. score based CUSUM, and CounterSignal.

**Drift among the patients who get treated** (from month 5, events among alerted patients become about three times as common, and the model's true O/E rises from about 0.98 to 1.35): share of deployments in which the first alarm came after the change began, and the median month of that alarm, counting the first month of the change as month 1.

<!-- tables:drift -->
| Dataset, model | Standard | Untreated only | Feng et al. | CounterSignal |
|---|---|---|---|---|
| Diabetes, logistic | 63.0% (month 3) | 88.0% (month 4) | 43.5% (month 5) | 100.0% (month 2) |
| Diabetes, boosting | 64.5% (month 3) | 81.0% (month 5) | 39.5% (month 4) | 99.5% (month 2) |
| Sepsis, logistic | 16.0% (month 3) | 72.5% (month 5) | 70.0% (month 4) | 99.5% (month 2) |
| Sepsis, boosting | 22.0% (month 3) | 61.5% (month 5) | 73.0% (month 4) | 99.5% (month 2) |
<!-- /tables -->

**Alarms when the model has not changed** (planned rate 5%):

<!-- tables:alarms_stable -->
| Dataset, model | Standard | Untreated only | Feng et al. | CounterSignal |
|---|---|---|---|---|
| Diabetes, logistic | 95.0% | 7.5% | 6.5% | 4.5% |
| Diabetes, boosting | 96.0% | 4.0% | 8.0% | 9.0% |
| Sepsis, logistic | 100.0% | 5.5% | 6.5% | 8.5% |
| Sepsis, boosting | 100.0% | 9.0% | 5.0% | 6.5% |
<!-- /tables -->

What this shows, and what it does not:

* CounterSignal caught drift among treated patients in at least 99.5% of deployments in every configuration, most often in the second month of the change (60.5% to 85.5% of deployments) and by then in 65.5% to 92.5%. The other monitors caught it in 16.0% to 88.0% of deployments, with a median first alarm in months 3 to 5 of the change; the rest alarmed before the change began (mostly standard monitoring) or not at all.
* With the model unchanged (true O/E 0.98 to 0.99, inside the acceptable range), it alarmed in 4.5% to 9.0% of deployments, similar to the untreated only and Feng et al. monitors. That is above the planned 5% in three of four configurations, up to 9.0%, which with 200 deployments is about 2.6 standard errors above it, so the 5% rate is not fully held. Standard monitoring alarmed in 95.0% to 100.0%.
* When treatment also depends on a factor the monitor is not given (1.0 log odds per standard deviation, twice the logged factor's effect), its alarm rate rose to 12.0% on the sepsis data. Logging every driver of treatment is the method's main requirement.
* Reweighting costs precision: single months are often undecided, and the cumulative alarm carries the evidence.

The treatment process is simulated, so this is evidence that the method works as designed on real case mix and outcomes, not a validation in a live deployment. Full design, every scenario, monthly estimate accuracy and limits: [docs/benchmark.md](docs/benchmark.md). Estimators and assumptions: [docs/method.md](docs/method.md).

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
| `src/countersignal/baselines/` | independent implementation of the Feng et al. score based CUSUM |
| `src/countersignal/datasets/` | UCI Diabetes and Sepsis loaders and cohort rules |
| `benchmarks/` | semi-synthetic benchmark, result aggregation and the tables in these docs |
| `results/` | benchmark outputs committed by the workflow |
| `dashboard/` | Streamlit app |
| `docs/` | method, benchmark case study, release notes |
| `tests/` | known truth tests for estimators, alarm calibration, edge cases, loaders and benchmark truth |

## Reproducibility

The *Data and benchmark* workflow downloads both archives from UCI and checks each against the SHA-256 in its manifest, failing if a checksum ever changes. The sepsis license statement was read from the UCI page by the workflow; the diabetes license (CC BY 4.0) was recorded when the archive was first added and matches the UCI page. The workflow runs the four benchmark configurations as parallel jobs and commits the combined results. *Quality checks* runs Ruff, the test suite and `benchmarks/tables.py --check` on every push to main and every pull request, so the generated tables in this README and in docs/benchmark.md match the committed results. Figures quoted in the prose are checked by hand.

## Data and citation

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J. Licensed CC BY 4.0.

Chicco, D., & Jurman, G. (2020). Sepsis Survival Minimal Clinical Records [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53C8N. Licensed CC BY 4.0. Introductory paper: Survival prediction of patients with sepsis from age, sex, and septic episode number alone, Scientific Reports 10, 17156 (2020).

Copies of both archives are kept in `data/raw/`, with checksums and license statements in `data/manifest*.json`.

Method references: Feng et al., *Monitoring machine learning (ML) based risk prediction algorithms in the presence of confounding medical interventions* (arXiv:2211.09781); Keogh and van Geloven, *Prediction under interventions: evaluation of counterfactual performance using longitudinal observational data* (Epidemiology, 2024).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
