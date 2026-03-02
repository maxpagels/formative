Estimands: ATE, ATT, and LATE
==============================

Every causal estimator targets a specific *estimand*. Choosing the wrong estimator for your question gives a valid answer to the wrong
question. The three estimands formative works with are ATE, ATT, and LATE.

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
e.g., "what would happen if we rolled this programme out to everyone?"

**Methods that estimate it:**

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
had they *not* been treated — a counterfactual that is never directly observed.

**When it makes sense:** When you care specifically about the effect for those who
self-selected into treatment, or when the treated group is the policy-relevant population —
e.g., "did the training programme benefit the workers who enrolled?"

**Methods that estimate it:**

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
by :math:`Z`. Units whose treatment is unaffected by the instrument — *always-takers* (always
treated regardless of :math:`Z`) and *never-takers* (never treated regardless of :math:`Z`)
— contribute no identifying variation. Only *compliers* — units who take treatment when
:math:`Z = 1` and not when :math:`Z = 0` — drive the IV estimate.

**When it makes sense:** When a clean instrument is available and you are willing to
interpret the result as the effect for compliers. If compliers are representative of the
broader population, LATE ≈ ATE. If not, the LATE may be very different from the ATE.

**Methods that estimate it:**

- **IV / 2SLS** — the Wald estimator (reduced form divided by first stage) identifies the
  LATE under the standard IV assumptions: relevance, exclusion restriction, independence,
  and monotonicity.

**LATE vs ATE vs ATT:** LATE is the narrowest estimand. It applies only to the complier
subpopulation, which is typically latent (you cannot directly observe who the compliers are).
Whether the LATE generalises depends on how similar compliers are to the rest of the
population — a substantive question that cannot be answered from the data alone.

----

Why not all methods can estimate ATE
--------------------------------------

If your goal is the ATE, you cannot freely substitute any estimator and expect to get it.
Each method's design commits it to a particular estimand, and that commitment is not a
limitation of the implementation — it is baked into the identification strategy itself.

**Propensity Score Matching targets ATT by construction.**
Nearest-neighbour matching finds a control unit for each *treated* unit. The comparison is
always anchored to the treated group: you are asking what would have happened to the treated
units if they had not been treated. There is no symmetric counterpart for untreated units,
so the estimator has nothing to say about what treatment would have done to them. To recover
ATE from matching you would need to match in both directions — finding a treated twin for
every control unit as well — which doubles the matching problem and requires common support
across the full covariate distribution, an assumption that is much harder to satisfy.

**DiD targets ATT because the counterfactual is group-specific.**
DiD uses the control group's time trend as a stand-in for what the treated group would have
experienced absent treatment. This counterfactual is constructed only for the treated group.
The parallel trends assumption says the two groups would have trended together, but it says
nothing about what the treatment effect would be for the control group if they had been
treated. The DiD interaction coefficient therefore recovers ATT, not ATE.

**IV recovers LATE, not ATE, because the instrument cannot reach all units.**
The instrument moves some units into (or out of) treatment but leaves others unaffected.
Always-takers are treated regardless of the instrument; never-takers are untreated regardless.
Neither group reveals anything about their treatment effect through the instrument's variation,
so they are effectively invisible to 2SLS. Recovering ATE would require knowing the treatment
effect for these groups too — which demands either additional instruments, strong parametric
assumptions about effect homogeneity, or extrapolation beyond what the data support.

**The practical implication.**
If you need ATE, use OLS (with sufficient covariate adjustment) or run an RCT. If those are
not feasible — because randomisation is impossible, or because unobserved confounders make
OLS unreliable — then you are in a setting where ATE is likely *not identified* from the
available data. Matching, DiD, and IV are not fallbacks that recover ATE under weaker
assumptions; they answer a different, more limited question. Recognising which question each
method answers is the first step to using them correctly.
