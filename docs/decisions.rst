Decisions
=========

Causal estimation answers "does X cause Y, and by how much?" but practitioners
usually need to go one step further: *should we act on this estimate?*

In formative, given a causal estimate and its uncertainty, you can run a decision
analysis to determine if a treatment is worthwhile. Simply estimating the causal effect
is not enough: it may be uncertain, or the cost of treatment may outweigh any benefit.

How it works
------------

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

Example
-------

Consider a job-training programme. We estimate the causal effect of training on
earnings using OLS with a DAG that encodes family background as a confounder:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from formative import DAG, OLSObservational

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

Value of information
--------------------

When a decision is *not* already highly confident, it is useful to know how much
more data would be needed to reach a target confidence level. Call
``value_of_information()`` on the report:

.. code-block:: python

   print(decision.value_of_information(target_confidence=0.99))

If the decision is already at or above the target confidence, formative reports
that no additional data is needed. Otherwise, it reports:

* how much the standard error on net benefit would need to shrink, and
* approximately how many times larger the sample would need to be.

This frames the cost of uncertainty concretely: collecting more data has a price,
and the value of information tells you whether that price is worth paying before
you commit to a decision.

Philosophy
----------

The decision layer is deliberately simple. It does not attempt to model complex
decision structures such as thresholds, multiple treatments, or dynamic policies.
It simply translates a causal estimate and its uncertainty into a binary decision:
treat or don't treat.

Most packages do not include such functionality, because decision-making is
inherently complex. formative does, simply because we believe that an attempt at numerical
decision analysis is better than none.

Note that a ``robust=False`` result is not a failure. It is an honest statement that the
data, as collected, cannot yet discriminate between two different actions. That is
valuable information.

API reference
-------------

.. autoclass:: formative.DecisionReport
   :members:
