Estimands: ATE, ATT, LATE etc.
==============================

Every causal estimator targets a specific *estimand*. Choosing the wrong estimator for your question gives a valid answer to the wrong
question. The four estimands formative works with are ATE, ATT, LATE, and LATE at the cutoff. One method can only target one estimand.

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
   * - **LATE at the cutoff**
     - Local Average Treatment Effect at the cutoff
     - :class:`~formative.RDD`

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
:math:`Z = 1` and not when :math:`Z = 0` — contribute to the estimate.

**When it makes sense:** When a clean instrument is available and you are willing to
interpret the result as the effect for compliers. If compliers are representative of the
broader population, LATE ≈ ATE. If not, the LATE may be very different from the ATE.

That is why IV is said to recover LATE, not ATE. In order to get ATE, you would need to know
the treatment effect for never-takers and always-takers too, which is not possible without
additional assumptions.

**Methods that estimate LATE:**

- **IV / 2SLS** — the Wald estimator (reduced form divided by first stage) identifies the
  LATE under the standard IV assumptions: relevance, exclusion restriction, independence,
  and monotonicity.

**LATE vs ATE vs ATT:** LATE is the narrowest estimand. It applies only to the complier
subpopulation, which is typically latent (you cannot directly observe who the compliers are).
Whether the LATE generalises depends on how similar compliers are to the rest of the
population. This cannot be answered from the data alone.

----

LATE at the cutoff: Local Average Treatment Effect at the cutoff
-----------------------------------------------------------------

**What it answers:** Among units just at the threshold of the running variable, what is
the effect of crossing from one side to the other?

.. math::

   \text{LATE at cutoff} = \lim_{\epsilon \to 0^+} \mathbb{E}[Y(1) - Y(0) \mid c \le X < c + \epsilon] - \mathbb{E}[Y(1) - Y(0) \mid c - \epsilon < X < c]

More intuitively: as the running variable :math:`X` approaches the cutoff :math:`c` from
either side, what is the jump in expected outcome?

RDD identifies this by fitting separate linear regressions on each side of the cutoff and
measuring the discontinuous jump at :math:`c`. Crucially, treatment is assigned deterministically
by the rule :math:`X \ge c`, so near the cutoff, units on either side are effectively comparable —
they differ only in whether they just cleared the threshold. This local exchangeability is what
makes identification possible without randomisation.

**When it makes sense:** When treatment is assigned by a threshold rule on a continuous running
variable — test score cutoffs, income eligibility limits, age thresholds — and you want to
know the causal effect for units near that threshold.

**The key limitation:** The estimate is *local*. It applies only to units near the cutoff,
not to units far from it. Whether the effect generalises to the rest of the distribution depends
on how much treatment effects vary with the running variable, which cannot be tested from the
data alone. If the running variable is a test score and the cutoff is the 60th percentile,
the LATE at the cutoff says nothing about the effect for units at the 30th or 90th percentile.

**LATE at the cutoff vs LATE (IV):** Both are "local" effects, but in different senses.
IV's LATE is local to compliers — a latent subpopulation defined by their response to the
instrument. RDD's LATE at the cutoff is local to a *region* of the running variable — units
near the threshold. If the bandwidth is wide, more units are included but the local
exchangeability assumption becomes harder to justify. If the bandwidth is narrow, the
assumption is more defensible but the estimate has higher variance.

**Methods that estimate LATE at the cutoff:**

- **RDD** — fits a local linear regression on both sides of the cutoff. The coefficient on
  the treatment indicator gives the jump in outcome at the threshold, controlling for the
  slope of the running variable separately on each side.
