Decisions
=========

Causal estimation answers "does X cause Y, and by how much?" but practitioners
usually need to go one step further: *should we act on this estimate?* Simply
estimating the causal effect is not enough: it may be uncertain, or the cost of
treatment may outweigh any benefit.

formative's decision layer answers the question at three levels of granularity,
each built on the same two numbers — the cost per unit of treatment applied and
the benefit per unit of improvement in the outcome:

1. **Should we treat at all?** ``result.decide(cost, benefit)`` — one
   cost-benefit call on the average effect.
2. **Should we treat each segment?** ``result.decide_by_group(cost, benefit)``
   — the same call per level of a pre-defined ``effect_modifier``.
3. **Whom should we treat?** ``result.learn_policy(...)`` — learns the
   segmentation itself: given several candidate features, it finds the
   treatment rule that maximises net benefit, and reports an honest estimate
   of what that rule is worth.

Should we treat at all?
-----------------------

Every result object exposes a ``.decide(cost, benefit)`` method. Pass the average cost
per unit of treatment applied and the monetary (or utility) value of one unit of
improvement in the outcome. formative returns a :class:`~formative.DecisionReport`
that answers three questions:

1. **What is the net benefit?** ``net_benefit = effect × benefit − cost``. If
   positive, treating is expected to be worthwhile.
2. **How confident are we?** ``p_beneficial`` is the probability that the true
   net benefit is positive, derived by treating the causal estimate as normally
   distributed around its point estimate with the reported standard error.
3. **Is the decision robust?** ``robust`` is ``True`` when the optimal decision
   (treat vs. don't treat) is the same at both ends of the 95% confidence
   interval. A fragile decision that flips within the CI is a signal
   that the estimate is too uncertain to act on without more data.

Consider a job-training programme. We estimate the causal effect of training on
earnings using OLS with a DAG that encodes family background as a confounder:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from formative.causal import DAG, OLSObservational

   rng = np.random.default_rng(0)
   N = 2_000
   background = rng.normal(size=N)
   training   = 0.6 * background + rng.normal(size=N)
   earnings   = 3.0 * training + 1.2 * background + rng.normal(size=N)

   df  = pd.DataFrame({"background": background, "training": training, "earnings": earnings})
   dag = DAG()
   dag.assume("background").causes("training", "earnings")
   dag.assume("training").causes("earnings")

   result = OLSObservational(dag, treatment="training", outcome="earnings").fit(df)

In the above example, the point estimate of the causal effect of training on earnings is around 3.0,
meaning that that for each unit increase in training, we expect a 3.0 unit increase in earnings.
Now suppose rolling out the programme costs $8 per participant, and each unit of
earnings increase is worth $15 in lifetime value. The expected net benefit per unit for training
is around ``3.0 × 15 − 8 = 37``, i.e. we expect to gain $37 for every unit of training applied.
But how confident are we in that estimate? And is it robust to estimation error?

.. code-block:: python

   decision = result.decide(cost=8, benefit=15)
   print(decision)

.. code-block:: text

   Decision Analysis: training → earnings
   ──────────────────────────────────────────────────
     Cost per unit of treatment   :     8.0000
     Benefit per unit of outcome  :    15.0000

     Net benefit (point estimate) :    +36.9958
     Net benefit 95% CI           : [+35.3981, +38.5935]

     Optimal decision             : treat
     Decision confidence          :    100.0%
     Robust to estimation error   : Yes — decision is stable across 95% CI

Game-theoretic robustness
~~~~~~~~~~~~~~~~~~~~~~~~~

``decide()`` uses expected value maximisation: it picks "treat" when
``effect × benefit − cost > 0``. This is the right rule when you want to
maximise the average outcome, but it is silent about risk attitude — it
treats a certain $37 gain and a 50/50 gamble between $0 and $74 identically.

The ``robust`` flag is a first step toward robustness: it checks whether
the decision flips anywhere inside the 95% confidence interval. But it
only returns ``True`` or ``False``, and it is implicitly using the most
conservative possible standard (the CI bounds).

For finer control, call ``to_outcomes()`` on the report and pass the result
to any rule in ``formative.game``:

.. code-block:: python

   from formative.game import maximin, minimax, hurwicz

   decision = result.decide(cost=8, benefit=15)
   outcomes = decision.to_outcomes()
   # {
   #   "treat":       {"pessimistic": ..., "expected": ..., "optimistic": ...},
   #   "don't treat": {"pessimistic": 0.0, "expected": 0.0, "optimistic": 0.0},
   # }

   maximin(outcomes).solve()            # best worst-case
   minimax(outcomes).solve()            # minimise maximum regret
   hurwicz(outcomes, alpha=0.3).solve() # weighted pessimism–optimism

By default, the three scenarios correspond to the 10th, 50th, and 90th
percentiles of the net-benefit sampling distribution (assumed normal with
the se derived from the 95% CI). The ``"don't treat"`` payoff is 0 in
every scenario — the status quo baseline.

You can supply your own scenario names and quantiles:

.. code-block:: python

   outcomes = decision.to_outcomes(
       scenarios={"bear": 0.05, "base": 0.50, "bull": 0.95}
   )

**Relationship to** ``robust``. ``robust=True`` is equivalent to
``maximin`` returning the same choice as ``optimal`` when the scenarios are
set to the CI bounds (quantiles 0.025 and 0.975). ``to_outcomes()``
generalises that check: different rules express different risk attitudes, and
you can dial in your own pessimism level via ``hurwicz(alpha=...)``.

Should we treat each segment?
-----------------------------

An average effect can hide groups where the treatment is worthless — or where
it pays for itself several times over. When an estimator was fitted with an
``effect_modifier`` (see :doc:`estimands`), ``decide_by_group(cost, benefit)``
runs the same cost-benefit analysis within each level of the modifier and
returns a mapping of level → :class:`~formative.DecisionReport`, so different
groups can receive different treatment decisions.

The remaining examples share one scenario: a company runs a randomised
coaching programme and measures employee retention. Coaching costs $2.5k per
participant and a unit of retention is worth $1k. It helps recent hires a lot
and everyone else not at all:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from formative.causal import DAG, RCT

   rng = np.random.default_rng(7)
   N = 5_000
   tenure    = rng.choice(["<2y", "2-5y", ">5y"], size=N)
   region    = rng.choice(["north", "south"], size=N)
   coaching  = rng.integers(0, 2, size=N)
   retention = 6.0 * coaching * (tenure == "<2y") + 2.0 * (tenure == ">5y") + rng.normal(size=N)

   df = pd.DataFrame({"tenure": tenure, "region": region, "coaching": coaching, "retention": retention})

   dag = DAG()
   dag.assume("coaching").causes("retention")
   dag.assume("tenure").causes("retention")
   dag.assume("region").causes("retention")

   result = RCT(dag, treatment="coaching", outcome="retention", effect_modifier="tenure").fit(df)
   for level, decision in result.decide_by_group(cost=2.5, benefit=1.0).items():
       lo, hi = decision.net_benefit_ci
       print(f"{level:>5}: {decision.optimal:<12} net benefit {decision.net_benefit:+.2f}  [{lo:+.2f}, {hi:+.2f}]")

.. code-block:: text

    2-5y: don't treat  net benefit -2.48  [-2.57, -2.38]
     <2y: treat        net benefit +3.42  [+3.33, +3.52]
     >5y: don't treat  net benefit -2.57  [-2.66, -2.47]

Coaching everyone would burn money on two-thirds of the workforce; coaching
recent hires is clearly worthwhile. The limitation is that *you* had to pick
``tenure`` as the segmentation. That is what the next level removes.

Whom should we treat? Learning a policy
---------------------------------------

``learn_policy()`` learns the segmentation itself: given several candidate
features, it finds the treatment rule that maximises net benefit and reports
an honest estimate of what that rule is worth.

The output is deliberately a *shallow decision tree* — e.g. "treat if
tenure = <2y" — because a rule a stakeholder can read, audit, and ship is
worth more than an opaque scoring model.

Policy learning is currently available on :class:`~formative.causal.RCTResult`:
with randomised treatment the machinery needs no propensity model, so there is
one less thing to get wrong.

How it works
~~~~~~~~~~~~

``learn_policy()`` implements doubly robust policy learning in the style of
Athey & Wager (2021):

1. **Score every unit.** Each unit gets a cross-fitted AIPW score — an
   unbiased estimate of its individual treatment effect. Outcome models are
   fit by OLS on the candidate features over four folds and evaluated on the
   fifth, so no unit is scored by a model that saw it.
2. **Convert to net benefit.** A unit's score becomes
   ``benefit × score − cost``: the estimated net gain of treating that unit.
3. **Search all shallow trees.** An exhaustive search over trees of depth
   ``max_depth`` (1 or 2), splitting on levels of the candidate features,
   finds the rule that maximises total net benefit. Ties prefer simpler trees.
4. **Estimate the value honestly.** Selecting the best-looking rule and then
   scoring it on the same data would overstate its value (a winner's curse).
   Instead, each unit is evaluated under a rule learned *without its fold*,
   and the reported ``value`` is the average advantage over the **best
   constant policy** — treating everyone or no one, whichever is better. A
   policy that learned nothing therefore reads as ≈ 0, even when treating
   everyone is profitable.

Because the tree may split on any level of any candidate feature, features
must be discrete: bin continuous columns before passing them in. Candidate
features face the same DAG validation as effect modifiers — each must cause
the outcome and must not be a descendant of the treatment (targeting on a
mediator is not actionable at assignment time).

Example
~~~~~~~

Continuing the coaching scenario, we now hand the learner *both* candidate
features instead of choosing a segmentation ourselves:

.. code-block:: python

   result = RCT(dag, treatment="coaching", outcome="retention").fit(df)
   policy = result.learn_policy(df, modifiers=["tenure", "region"], cost=2.5, benefit=1.0, max_depth=2)
   print(policy.summary())

.. code-block:: text

   Learned Policy: coaching → retention
     Estimand: policy value vs best constant policy (doubly robust)
   ──────────────────────────────────────────────────
     Policy rule
     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
       treat if tenure = <2y
       otherwise don't treat

     Value vs constant    :    +1.1293 per unit
     Std. error           :     0.0281
     95% CI               : [+1.0742, +1.1843]
     Coverage             :      33.0%
     Cost / benefit       : 2.5 / 1
     N                    :       5000

The learner recovered the true rule — it treats only recent hires, a third of
the workforce — and did not split on ``region``, which carries no signal.
The value line says targeting adds about $1.13k per employee over the best
one-size-fits-all option, with a confidence interval well clear of zero.

The rule can be applied directly to new data:

.. code-block:: python

   new_hires = pd.DataFrame({"tenure": ["<2y", ">5y"], "region": ["north", "south"]})
   policy.assign(new_hires)   # True, False

Combining features
~~~~~~~~~~~~~~~~~~

The example above only needed one feature, but the point of passing several
``modifiers`` is that a depth-2 tree can express an *interaction*. Suppose
coaching is delivered in person and only the north has coaches on site — so
it only helps recent hires in the north:

.. code-block:: python

   rng = np.random.default_rng(7)
   tenure    = rng.choice(["<2y", "2-5y", ">5y"], size=N)
   region    = rng.choice(["north", "south"], size=N)
   coaching  = rng.integers(0, 2, size=N)
   helped    = (tenure == "<2y") & (region == "north")
   retention = 7.0 * coaching * helped + 2.0 * (tenure == ">5y") + rng.normal(size=N)
   df = pd.DataFrame({"tenure": tenure, "region": region, "coaching": coaching, "retention": retention})

   result = RCT(dag, treatment="coaching", outcome="retention").fit(df)
   policy = result.learn_policy(df, modifiers=["tenure", "region"], cost=2.5, benefit=1.0, max_depth=2)
   print(policy.summary())

.. code-block:: text

   Learned Policy: coaching → retention
     Estimand: policy value vs best constant policy (doubly robust)
   ──────────────────────────────────────────────────
     Policy rule
     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
       treat if tenure = <2y and region ≠ south
       otherwise don't treat

     Value vs constant    :    +0.7533 per unit
     Std. error           :     0.0293
     95% CI               : [+0.6959, +0.8108]
     Coverage             :      16.6%
     Cost / benefit       : 2.5 / 1
     N                    :       5000

The learned rule combines both features and treats only the one-sixth of
employees where coaching pays for itself. This is also where comparing depths
earns its keep — a depth-1 tree cannot express the interaction, so it falls
back to treating all recent hires and burns money on the southern half:

.. code-block:: python

   result.learn_policy(df, modifiers=["tenure", "region"], cost=2.5, benefit=1.0, max_depth=1).value
   # +0.34 per unit — versus +0.75 at depth 2

When the deeper policy's value is clearly higher (as here), ship it; when the
two are within noise of each other, prefer the simpler rule.

Policies with several treated groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A policy is not limited to one treated group: ``rules`` prints one
``treat if …`` line for every path through the tree that ends in *treat*.
Here training pays off for two disjoint groups — everyone in segment 0, and
segment-2 employees in the north:

.. code-block:: python

   rng = np.random.default_rng(3)
   N = 6_000
   segment  = rng.integers(0, 3, size=N)
   region   = rng.choice(["north", "south"], size=N)
   training = rng.integers(0, 2, size=N)
   effect   = 4.0 * (segment == 0) + 5.0 * ((segment == 2) & (region == "north"))
   earnings = effect * training + 0.5 * segment + rng.normal(size=N)
   df = pd.DataFrame({"segment": segment, "region": region, "training": training, "earnings": earnings})

   dag = DAG()
   dag.assume("training").causes("earnings")
   dag.assume("segment").causes("earnings")
   dag.assume("region").causes("earnings")

   result = RCT(dag, treatment="training", outcome="earnings").fit(df)
   policy = result.learn_policy(df, modifiers=["segment", "region"], cost=1.0, benefit=1.0, max_depth=2)
   print(policy.summary())

.. code-block:: text

   Learned Policy: training → earnings
     Estimand: policy value vs best constant policy (doubly robust)
   ──────────────────────────────────────────────────
     Policy rule
     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
       treat if segment = 2 and region ≠ south
       treat if segment ≠ 2 and segment ≠ 1
       otherwise don't treat

     Value vs constant    :    +0.5119 per unit
     Std. error           :     0.0219
     95% CI               : [+0.4690, +0.5548]
     Coverage             :      50.4%
     Cost / benefit       : 1 / 1
     N                    :       6000

Each line is one route to a treat leaf, so conditions read in tree order and
may arrive in complementary form: the second line is the segment-0 group,
reached by excluding segments 2 and 1. Whatever the phrasing, ``assign()``
applies the rule exactly — here it treats precisely the two profitable groups.

Refuting a learned policy
~~~~~~~~~~~~~~~~~~~~~~~~~

As with every estimator in formative, the result can be stress-tested with
``policy.refute(df)``:

- **Placebo modifiers** — every candidate feature column is permuted and the
  learner re-run. With the features scrambled there is nothing to target, so
  the placebo policy's value should collapse to ≈ 0.
- **Random modifier** — a pure-noise feature is added to the candidates. The
  honest value should not improve: if noise helps, the value estimate cannot
  be trusted.

.. code-block:: python

   print(policy.refute(df).summary())

.. code-block:: text

   Policy Refutation Report: coaching → retention
   ──────────────────────────────────────────────────
     [PASS]  Placebo modifiers: placebo policy value = +0.0000 (SE 0.0000)  Not significantly positive.
     [PASS]  Random modifier: value shifted by +0.0000  (≤ 1 SE = 0.0281)  Noise adds no value, as expected.

     All checks passed.

Practical notes
~~~~~~~~~~~~~~~

- ``max_depth`` is the regulariser, and it is capped at 2 — deeper trees stop
  being auditable. Fit a depth-1 and a depth-2 policy and compare their
  ``value``: if the deeper tree isn't clearly worth more, ship the simpler one.
- The ``value`` describes the learning *procedure*, evaluated out-of-fold;
  the printed rule is re-learned on the full sample. This is what makes the
  estimate honest, at the cost of a slight mismatch between the rule you see
  and the folds behind the number.
- The treatment must be binary 0/1, and each arm needs at least 10 units for
  cross-fitting (in practice you want far more).

Philosophy
----------

The decision layer is deliberately simple. Each level translates a causal
estimate and its uncertainty into an auditable action: a treat/don't-treat
call, one call per segment, or a shallow readable rule. It does not attempt
opaque scoring models, multi-valued treatments, or dynamic/sequential
policies.

Most packages do not include such functionality, because decision-making is
inherently complex. formative does, simply because we believe that an attempt at numerical
decision analysis is better than none.

Note that a ``robust=False`` result is not a failure. It is an honest statement that the
data, as collected, cannot yet discriminate between two different actions. That is
valuable information.

API reference
-------------

DecisionReport
~~~~~~~~~~~~~~

.. autoclass:: formative.causal.DecisionReport
   :members:

PolicyResult
~~~~~~~~~~~~

.. autoclass:: formative.causal.PolicyResult
   :members:

PolicyNode
~~~~~~~~~~

.. autoclass:: formative.causal.PolicyNode
   :members:

PolicyRefutationReport
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: formative.causal.PolicyRefutationReport
   :members:
   :inherited-members:
