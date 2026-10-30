# Method, assumptions and limits

## The problem

A risk model is deployed to flag patients for an intervention. Once clinicians act on its alerts, the outcomes recorded for flagged patients are no longer the outcomes the model was built to predict. If the intervention works, flagged patients have fewer events than predicted, and a monitor that compares predictions with recorded outcomes concludes the model over predicts. The more effective the intervention, the larger this apparent miscalibration.

Feng and colleagues (2022) call these confounding medical interventions. They show that even the obvious fix, monitoring only untreated patients, can inflate false alarms, because who goes untreated depends on the model's own score. The untreated are also mostly low risk, so a change concentrated among high risk patients, the ones clinicians treat, is hard to see in them.

## What CounterSignal estimates

The target is the model's performance against the **untreated outcome**: what would have happened to each patient under standard care. That is what the model was validated to predict, so it is the right yardstick for "is the model still working".

"Calibrated" here is absolute: observed events equal the events the score predicts (O/E = 1). It is not "unchanged since validation". A model whose validation O/E was 1.15 will be reported as under predicting (15% more events than predicted) from the first month, and a model that drifts from 0.85 to 1.00 is reported as becoming calibrated. The validation data are used for the AUC verdict, the case mix signals and the doubly robust outcome model, not as the calibration target.

For each monitoring window (normally a month), CounterSignal reports:

* **Observed over expected (O/E)**: events divided by predicted events. 1.00 is calibrated; above 1 the model under predicts.
* **AUC**: how well the score ranks patients who have the event above those who do not.
* **Calibration intercept** (log odds offset with the slope fixed at 1) and **calibration slope** (from a joint logistic recalibration).
* **Brier score**.

Observed/expected and AUC come in three versions so the difference is visible; the calibration intercept and slope, the Brier score and the doubly robust estimate are computed for the CounterSignal view only.

| View | Patients | Weights | What it measures |
|---|---|---|---|
| Standard | everyone | none | performance against recorded outcomes, which mixes in treatment |
| Untreated only | untreated | none | performance among the kind of patient who goes untreated |
| CounterSignal | untreated | 1 / P(untreated given covariates) | performance against the untreated outcome in the whole deployed population |

## How the weights work

CounterSignal fits a logistic regression of the logged treatment on the model's score (log odds) and the covariates that influenced treatment, refitted each month on all deployment data seen so far. Each untreated patient gets weight 1 / P(untreated given covariates); treated patients get weight 0. A patient who was unlikely to be left untreated stands in for the similar patients who were treated. This is inverse probability weighting, the approach used for counterfactual performance by Keogh and van Geloven (2024) and in the R packages ipeval and cfperformance, applied here month after month.

Weights are capped at 50 (a floor of 0.02 on the probability) and the number of capped patients is reported, because heavy capping means positivity is in doubt. The effective sample size of the weights is reported for every window.

A doubly robust O/E is also reported. It combines the weights with an outcome model of the untreated outcome given the score and covariates. Because that outcome model is fitted on the reference period, it is only correct while nothing has changed; once the model drifts, the doubly robust estimate is protected only by the treatment model, exactly like the weighted estimate. It is reported for comparison and is not used in any verdict or in the alarm.

## Verdicts that admit uncertainty

Observed/expected, AUC, Brier score and the doubly robust O/E get 95% bootstrap percentile intervals: patients are resampled within the window and the treatment model is refitted on each resample, so the intervals include the uncertainty in the weights. The calibration intercept and slope are point estimates. The O/E verdict compares the interval with an acceptable range, 0.80 to 1.25 by default:

* **stable** when the whole interval is inside the range,
* **drift** when the whole interval is outside it,
* **not enough evidence yet** otherwise.

For that last case CounterSignal estimates how large a single month would have to be for the interval to reach the nearer edge of the range, assuming the point estimate holds and the interval narrows with the square root of the effective sample. It is reported in patients, next to the month's actual size. At that size the interval only touches the edge, so a month that large settles the question about half the time; it is a guide to scale, not a guarantee. A small hospital gets "not enough evidence yet" with that figure instead of a coin flip alarm. The AUC verdict works the same way against the reference AUC minus a tolerance (0.05 by default).

## The cumulative alarm

Single months can be noisy, so evidence is accumulated across months with a two sided CUSUM.

1. Each month's statistic is the weighted sum of (outcome minus prediction) over untreated patients.
2. The null hypothesis is that the score is the true untreated risk given the logged covariates: each untreated outcome is a Bernoulli draw with probability equal to the score. CounterSignal simulates that null distribution by Monte Carlo (2,000 draws) from the month's own scores and weights and converts the observed statistic to a normal score through its mid rank. There is no normal approximation, which keeps the alarm valid in small windows; the weights are treated as fixed, so the uncertainty of the treatment model is not part of this step. This null is stronger than calibration in the large. A model that is calibrated overall but not within levels of a covariate that drives treatment adds variance the simulation leaves out, which raises the false alarm rate above the planned level. In the benchmark, where the real models are only approximately calibrated within covariate levels, the alarm ran at 4.5% to 9.0% against a planned 5%.
3. The CUSUM adds those scores with allowance k = 0.5. The alarm limit is set by simulation so that a calibrated model triggers any alarm over the planned horizon (12 months by default) with probability 5%. Monitoring beyond the horizon exceeds that rate, and the monitor warns when it does.

Months are processed in the sorted order of their labels, so labels must sort in time order (2026-01, 2026-02 or M01, M02). A month is evaluated only once all of its outcomes are recorded; until then it is "awaiting outcomes". A month in which every patient was treated is reported as "no untreated patients".

The same rule is applied to the standard and untreated only views, so comparisons between them are like for like.

## Early signals while outcomes are pending

Outcomes such as readmission are known only weeks after discharge. For months still awaiting outcomes, CounterSignal still reports the population stability index (PSI) of the score against the reference data, and the covariate with the largest PSI. Binary and other discrete variables (ten or fewer values) are compared level by level. PSI says the population changed, not that performance changed. The usual reading (below 0.1 little change, 0.1 to 0.25 moderate, above 0.25 large) is a convention, not a test.

## Assumptions

The corrected estimates are only as good as these:

1. **Everything that drove treatment is logged.** If clinicians treated on information that is not in the covariates and that also predicts the outcome, the weights cannot remove the bias. In practice the alert flag, the score and the clinical fields shown at the point of decision should be logged.
2. **Positivity.** Every kind of patient must have some chance of going untreated. If every alerted patient is treated, nothing in the data says what their untreated outcome would have been, and no method can recover it without strong modelling assumptions.
3. **The reference period was untreated by the model.** The reference data sets the AUC benchmark and fits the doubly robust outcome model, and must come from before the model influenced care.
4. **The treatment model is correctly specified.** Treatment that jumps at the alert threshold needs the alert flag as a covariate. (The doubly robust estimate does not relax this once the model has drifted; see above.)
5. **Outcomes are recorded for every patient** in a window once it closes. Loss to follow up that depends on treatment would need its own weights.

## What this is not

CounterSignal does not estimate whether the intervention works, does not rank hospitals, and does not decide whether to retrain. It reports whether the model still predicts what it was validated to predict, with honest uncertainty, so that people can decide.

## References

* Feng J, Gossmann A, Pennello G, Petrick N, Sahiner B, Pirracchio R. Monitoring machine learning (ML) based risk prediction algorithms in the presence of confounding medical interventions. arXiv:2211.09781. https://arxiv.org/abs/2211.09781
* Keogh RH, van Geloven N. Prediction under interventions: evaluation of counterfactual performance using longitudinal observational data. Epidemiology. 2024;35(3). https://pmc.ncbi.nlm.nih.gov/articles/PMC11332371/
* ipeval: https://survival-lumc.github.io/ipeval/ · cfperformance: https://christopherbboyer.com/cfperformance/
* Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN. Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. BioMed Research International. 2014;781670.
* Clore J, Cios K, DeShazo J, Strack B. Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5230J (CC BY 4.0)
* Chicco D, Jurman G. Survival prediction of patients with sepsis from age, sex, and septic episode number alone. Scientific Reports. 2020;10:17156. https://doi.org/10.1038/s41598-020-73558-3. Dataset: Sepsis Survival Minimal Clinical Records, UCI Machine Learning Repository, https://doi.org/10.24432/C53C8N (CC BY 4.0)
