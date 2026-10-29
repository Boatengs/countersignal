# Benchmark on the UCI Diabetes 130-US Hospitals records

Every figure on this page comes from `results/benchmark_summary.json` and `results/benchmark_runs.csv`, produced by the *Data and benchmark* workflow from the archive recorded in `data/manifest.json` (SHA-256 `f82ac129…112a`).

## Why a semi-synthetic benchmark

To score a monitor you need to know the truth: how the deployed model actually performs in each month. Real deployments never reveal that, because once the model's alerts change care, the untreated outcome of treated patients is never observed. The benchmark gets around this by using real patients whose care was not influenced by any model, treating their recorded outcome as the untreated outcome, and simulating the deployment on top of it. The patients, features and readmissions are real. The alerts, the treatment decisions and the treatment effect are simulated, with the assumptions stated below.

## Data and cohort

| Step | Count |
|---|---|
| Encounters in the UCI file | 101,766 |
| Distinct patients (first encounter kept, lowest `encounter_id`) | 71,518 |
| After removing deaths and hospice discharges (disposition 11, 13, 14, 19, 20, 21) | 69,973 |
| Readmitted within 30 days | 6,277 (8.97%) |

The cohort rules follow the dataset's introductory paper (Strack et al., 2014). The file has no admission dates, so the order of encounters carries no meaning and months in the simulation are random draws.

## Model

A logistic regression on 24 standardized features predicts 30 day readmission: length of stay; counts of lab tests, procedures, medications and diagnoses; prior outpatient, emergency and inpatient visits; age; sex; emergency admission; discharge home; HbA1c tested; medication change; any diabetes medication; insulin; and eight primary diagnosis groups.

| Split | Patients | Use |
|---|---|---|
| Development | 20,991 | fit the model |
| Reference | 13,995 | validation before deployment: AUC 0.631, observed/expected 1.066 |
| Pool | 34,987 | source of the monthly deployment windows: AUC 0.618, observed/expected 0.986 |

The model is deliberately ordinary. CounterSignal monitors models; it does not need a strong one to show the effect.

## Simulated deployment

* 12 months per deployment; each month draws 2,000 distinct patients from the pool (150 in the small hospital scenario).
* Patients at or above the reference 90th percentile of risk (score 0.141) are alerted.
* Clinicians treat 70% of alerted and 5% of other patients, with the log odds of treatment rising by 0.5 per standard deviation of prior inpatient stays.
* Treatment prevents 40% of the readmissions it is applied to (sensitivity runs use 20% and 60%).
* The monitor's treatment model uses the logged alert flag, prior inpatient stays and age.
* Verdict range for observed/expected: 0.80 to 1.25. Alarm limit set for a 5% chance of any false alarm in 12 months.
* 200 deployments per scenario and effect size, 200 bootstrap resamples per month.

These treatment numbers are assumptions chosen to make the problem visible, not estimates of any real programme.

## Scenarios

| Scenario | What changes |
|---|---|
| Model unchanged | nothing; any alarm is false |
| Sicker case mix | from month 5, patients with more prior inpatient stays are oversampled; the model is unchanged |
| Rising readmissions among alerted patients | from month 5, readmitted patients are three times as likely to be drawn among alerted patients, so true risk rises in the group most likely to be treated |
| Small hospital | 150 patients a month; the model is unchanged |

## Results

Three monitors use the same data, the same cumulative alarm rule and the same limit:

* **Standard**: predictions against every patient's recorded outcome.
* **Untreated only**: treated patients dropped, no reweighting.
* **CounterSignal**: untreated patients weighted by the inverse probability of remaining untreated.

### False alarms when the model has not changed

Share of 12 month deployments in which the cumulative alarm fired at any point. The planned rate is 5%.

| Scenario | Standard | Untreated only | CounterSignal |
|---|---|---|---|
| Model unchanged | 96.5% | 10.0% | 5.5% |
| Sicker case mix | 100.0% | 12.5% | 1.5% |
| Small hospital | 11.5% | 6.5% | 3.0% |

With the model unchanged, standard monitoring raised a false alarm in almost every deployment, because prevented readmissions look like over prediction. Dropping treated patients removes most of the problem but still alarmed at twice the planned rate. CounterSignal stayed close to the planned 5%.

### Detecting real drift among the patients who get treated

| Measure | Standard | Untreated only | CounterSignal |
|---|---|---|---|
| Deployments where the change was caught | 57.0% | 65.0% | 97.5% |
| False alarm before the change began | 26.5% | 0.0% | 0.0% |
| Median month of first alarm after the change (interquartile range) | 5 (3 to 7) | 5 (4 to 6) | 3 (3 to 4) |

The rise in risk sits among alerted patients, most of whom are treated. Standard monitoring partly cancels it against prevented readmissions; the untreated only view barely sees those patients. CounterSignal caught the change in 195 of 200 deployments, about two months sooner, with no alarm before it began.

### Sensitivity to the treatment effect (model unchanged)

| Share of readmissions prevented | Standard | Untreated only | CounterSignal |
|---|---|---|---|
| 20% | 51.5% | 6.5% | 3.5% |
| 40% | 96.5% | 10.0% | 5.5% |
| 60% | 100.0% | 9.0% | 5.0% |

The more effective the intervention, the more often standard monitoring raises a false alarm. CounterSignal's false alarm rate does not depend on it.

## What the correction costs

Reweighting is not free. With the model unchanged and 2,000 patients a month:

* The weights leave about 1,217 effective patients a month.
* CounterSignal's monthly observed/expected was off from the truth by 0.064 on average, against 0.036 for the untreated only view (and 0.075 for standard monitoring).
* It returned "not enough evidence yet" in 9.2 of 12 months, against 4.8 for the untreated only view. Its 95% intervals contained the true value in 99.3% of months, so they are, if anything, conservative.

In other words, single months rarely settle the question at this volume, and the cumulative alarm is what carries the evidence. In the small hospital scenario all three monitors returned "not enough evidence yet" in nearly every month (11.7 of 12 for CounterSignal), which is the honest answer with about 106 effective patients a month.

## Limits of this benchmark

* Treatment, its effect and its dependence on covariates are simulated. Real treatment decisions may depend on information that is not logged, which would bias any weighting method.
* Every driver of treatment in the simulation is logged and given to the monitor. That is the best case for CounterSignal's main assumption.
* Months are random draws from one pool, so there is no real calendar drift; drift is introduced by resampling real records.
* Patients can appear in more than one month (never twice in the same month).
* One dataset, one model and one alert policy.

## Reproduce

```bash
pip install -e .
python benchmarks/run_benchmark.py --replicates 200 --n-boot 200
```

The archive must be at `data/raw/diabetes_130_us_hospitals.zip`; the workflow downloads it from UCI and checks it against `data/manifest.json`. The run took about 20 minutes on a GitHub hosted runner.
