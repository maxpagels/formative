Estimands: ATE, ATT, and LATE
==============================

Every causal estimator targets a specific *estimand*. Choosing the wrong estimator for your question gives a valid answer to the wrong
question. The three estimands formative works with are ATE, ATT, and LATE. One method can only target one estimand.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Estimand
     - Full name
     - Method
   * - **ATE**
     - Average Treatment Effect
     - :class:`~formative.OLSObservational`, :class:`~formative.RCT`
   * - **ATT**
     - Average Treatment Effect on the Treated
     - :class:`~formative.PropensityScoreMatching`, :class:`~formative.DiD`
   * - **LATE**
     - Local Average Treatment Effect
     - :class:`~formative.IV2SLS`

----

ATE: Average Treatment Effect
--------------------------------

**What it answers:** If we randomly assigned treatment to everyone in the population, what
would the average change in outcome be?

.. math::

   \text{ATE} = \mathbb{E}[Y(1) - Y(0)]

:math:`Y(1)` is the potential outcome under treatment and :math:`Y(0)` under control, for
the same unit. ATE averages this difference across *all units*, treated and untreated alike.

**When it makes sense:** When you want a policy-relevant effect for the whole population —
e.g., "what would happen if we rolled this programme out to everyone?". Note that this is
usually the question you want answered, but strictly speaking only OLS and RCT can answer it
in formative.

RCTs can do ATE because you are randomising the treatment, so the characteristics of the
treatment and control groups are assumed to be balanced. If the ATE is 1.5, the estimate
applies to the control group too, because there is no reason to assume the units in the control
group are systematically different from the treated group.

Assuming a perfect world where all confounders are observed, OLS can also theoretically do ATE. The units
in the treatment groups are not randomised, but if you adjust for all confounders that affect
both treatment and outcome, you can recover the ATE.

**Methods that estimate ATE:**

- **OLS Observational** — estimates ATE by adjusting for confounders identified via the
  backdoor criterion. Requires all relevant confounders to be observed.
- **RCT** — randomisation makes treated and control groups exchangeable, so the simple
  difference in means is an unbiased ATE estimate with no adjustment required.

----

ATT: Average Treatment Effect on the Treated
-----------------------------------------------

**What it answers:** Among units that actually received treatment, how much did treatment
change their outcome?

.. math::

   \text{ATT} = \mathbb{E}[Y(1) - Y(0) \mid \text{treated}]

ATT conditions on the treated group. It asks what the treated units would have experienced
had they *not* been treated. This is a counterfactual that is never directly observed.

**When it makes sense:** When you care specifically about the effect for those who
self-selected into treatment, or when the treated group is the policy-relevant population —
e.g., "did the training programme benefit the workers who enrolled?". Note that this is a
"narrower" question than ATE, and the answer may not generalise to the broader population.
However, in practice, ATT is often used to estimate ATE, even if theoretically it is not
the same thing.

Matching can do ATT but not ATE. For each treated unit, it finds one or more untreated units
that look similar on observables. This constructs the missing counterfactual, i.e.
what would the treated unit have experienced without treatment. In order to recover ATE,
you would have to do the same for untreated units. Finding treated matches for any given
control unit is hard; usually the treatment group is much smaller than the control group,
so finding a good match for every control unit becomes a much harder problem to solve.

The same restriction applies to DiD, but for a different reason. DiD constructs the
counterfactual for the treated group by using the control group's time trend.
The parallel trends assumption says the two groups would have trended together,
but it says nothing about what the treatment effect would be for the control group
if they had been treated. In order to get DiD to recover ATE, you would have to
assume that the treatment effect is the same for both groups, which is a strong assumption.
For example, say you launched an app in a few countries first. You probably launched assuming
the app would be more beneficial in those countries than in the others, so the ATT is likely
greater than the ATE.

**Methods that estimate ATT:**

- **Propensity Score Matching** — each treated unit is matched to the most similar control
  unit by propensity score, and the ATT is the average outcome difference across matched
  pairs.
- **DiD** — compares how outcomes changed over time for the treated group versus a control
  group. Under parallel trends, the common time trend cancels out, leaving the ATT as the
  interaction coefficient on group × time.

**ATE vs ATT:** When treatment is randomly assigned (as in an RCT), ATE = ATT, because the
treated group is a random draw from the population. In observational settings they can
differ — people who select into treatment often do so because the treatment is particularly
beneficial for them (positive selection), making ATT > ATE.

----

LATE: Local Average Treatment Effect
---------------------------------------

**What it answers:** Among units whose treatment status was *moved* by the instrument, what
was the effect of treatment?

.. math::

   \text{LATE} = \mathbb{E}[Y(1) - Y(0) \mid \text{complier}]

IV estimation with an instrument :math:`Z` isolates only the variation in treatment caused
by :math:`Z`. Only "compliers" — units who take treatment when
:math:`Z = 1` and not when :math:`Z = 0` — contribute to the estimate. That is why IV
recovers LATE, not ATE. In order to get ATE, you would need to know the treatment effect for
never-takers and always-takers too, which is not possible without additional assumptions.

**When it makes sense:** When a clean instrument is available and you are willing to
interpret the result as the effect for compliers. If compliers are representative of the
broader population, LATE ≈ ATE. If not, the LATE may be very different from the ATE.

**Methods that estimate LATE:**

- **IV / 2SLS** — the Wald estimator (reduced form divided by first stage) identifies the
  LATE under the standard IV assumptions: relevance, exclusion restriction, independence,
  and monotonicity.

**LATE vs ATE vs ATT:** LATE is the narrowest estimand. It applies only to the complier
subpopulation, which is typically latent (you cannot directly observe who the compliers are).
Whether the LATE generalises depends on how similar compliers are to the rest of the
population. This cannot be answered from the data alone.
