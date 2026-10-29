# Method, assumptions and limits

## The problem

A risk model is deployed to flag patients for an intervention. Once clinicians act on its alerts, the outcomes recorded for flagged patients are no longer the outcomes the model was built to predict. If the intervention works, flagged patients have fewer events than predicted, and a monitor that compares predictions with recorded outcomes concludes the model over predicts. The more effective the intervention, the larger this apparent miscalibration.

Feng and colleagues (2022) call these confounding medical interventions. They show that even the obvious fix, monitoring only untreated patients, can inflate false alarms, because who goes untreated depends on the model's own score. The untreated are also mostly low risk, so a change concentrated among high risk patients, the ones clinicians treat, is hard to see in them.

## What CounterSignal estimates

The target is the model's performance against the **untreated outcome**: what would have happened to each patient under standard care. That is what the model was validated to predict, so it is the right yardstick for "is the model still working".

For each monitoring window (normally a month), CounterSignal reports:

* **Observed over expected (O/E)**: events divided by predicted events. 1.00 is calibrated; above 1 the model under predicts.
* **AUC**: how well the score ranks patients who have the event above those who do not.
* **Calibration intercept and slope** from a logistic recalibration.
* **Brier score**.

Each comes in three versions so the difference is visible:

| View | Patients | Weights | What it measures |
|---|---|---|---|
| Standard | everyone | none | performance against recorded outcomes, which mixes in treatment |
| Untreated only | untreated | none | performance among the kind of patient who goes untreated |
| CounterSignal | untreated | 1 / P(untreated given covariates) | performance against the untreated outcome in the whole deployed population |

## How the weights work

CounterSignal fits a logistic regression of the logged treatment on the model's score (log odds) and the covariates that influenced treatment, refitted each month on all deployment data seen so far. Each untreated patient gets weight 1 / P(untreated given covariates); treated patients get weight 0. A patient who was unlikely to be left untreated stands in for the similar patients who were treated. This is inverse probability weighting, the approach used for counterfactual performance by Keogh and van Geloven (2024) and in the R packages ipeval and cfperformance, applied here month after month.

Weights are capped at 50 (a floor of 0.02 on the probability) and the number of capped patients is reported, because heavy capping means positivity is in doubt. The effective sample size of the weights is reported for every window.

A doubly robust O/E is also reported. It combines the weights with an outcome model fitted on the reference data and stays consistent if either the treatment model or the outcome model is right.

## Verdicts that admit uncertainty

Every metric gets a 95% bootstrap percentile interval (patients resampled within the window, weights held fixed). The O/E verdict compares the interval with an acceptable range, 0.80 to 1.25 by default:

* **stable** when the whole interval is inside the range,
* **drift** when the whole interval is outside it,
* **not enough evidence yet** otherwise.

For that last case CounterSignal estimates how many effectively untreated patients would settle the question, assuming the point estimate holds and the interval narrows with the square root of the sample. A small hospital gets "not enough evidence yet, about N more patients needed" instead of a coin flip alarm. The AUC verdict works the same way against the reference AUC minus a tolerance (0.05 by default).

## The cumulative alarm

Single months can be noisy, so evidence is accumulated across months with a two sided CUSUM.

1. Each month's statistic is the weighted sum of (outcome minus prediction) over untreated patients.
2. Under the hypothesis that the model is calibrated for the untreated outcome, each untreated outcome is a Bernoulli draw with probability equal to the score. CounterSignal simulates that null distribution exactly from the month's own patients (2,000 draws) and converts the observed statistic to a normal score through its mid rank. This keeps the alarm honest in small windows where normal approximations fail.
3. The CUSUM adds those scores with allowance k = 0.5. The alarm limit is set by simulation so that a calibrated model triggers any alarm over the planned horizon (12 months by default) with probability 5%.

The same rule is applied to the standard and untreated only views, so comparisons between them are like for like.

## Early signals while outcomes are pending

Readmission is known only 30 days after discharge. For months with no outcomes yet, CounterSignal still reports the population stability index (PSI) of the score and of each covariate against the reference data. PSI says the population changed, not that performance changed. The usual reading (below 0.1 little change, 0.1 to 0.25 moderate, above 0.25 large) is a convention, not a test.

## Assumptions

The corrected estimates are only as good as these:

1. **Everything that drove treatment is logged.** If clinicians treated on information that is not in the covariates and that also predicts the outcome, the weights cannot remove the bias. In practice the alert flag, the score and the clinical fields shown at the point of decision should be logged.
2. **Positivity.** Every kind of patient must have some chance of going untreated. If every alerted patient is treated, nothing in the data says what their untreated outcome would have been, and no method can recover it without strong modelling assumptions.
3. **The reference period was untreated by the model.** The reference data defines "calibrated" and must come from before the model influenced care.
4. **The treatment model is correctly specified,** or the outcome model is, for the doubly robust estimate. Treatment that jumps at the alert threshold needs the alert flag as a covariate.
5. **Outcomes are recorded for every patient** in a window once it closes. Loss to follow up that depends on treatment would need its own weights.

## What this is not

CounterSignal does not estimate whether the intervention works, does not rank hospitals, and does not decide whether to retrain. It reports whether the model still predicts what it was validated to predict, with honest uncertainty, so that people can decide.

## References

* Feng J, Gossmann A, Pennello G, Petrick N, Sahiner B, Pirracchio R. Monitoring machine learning (ML) based risk prediction algorithms in the presence of confounding medical interventions. arXiv:2211.09781. https://arxiv.org/abs/2211.09781
* Keogh RH, van Geloven N. Prediction under interventions: evaluation of counterfactual performance using longitudinal observational data. Epidemiology. 2024;35(3). https://pmc.ncbi.nlm.nih.gov/articles/PMC11332371/
* ipeval: https://survival-lumc.github.io/ipeval/ · cfperformance: https://christopherbboyer.com/cfperformance/
* Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN. Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. BioMed Research International. 2014;781670.
* Clore J, Cios K, DeShazo J, Strack B. Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5230J (CC BY 4.0)
