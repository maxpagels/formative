Policy Learning
===============

Causal estimation asks *does the treatment work?* Decision analysis asks
*is it worth applying?* Policy learning asks the last question in the chain:
**whom should we treat?**

formative's decision tools form a ladder:

1. ``result.decide(cost, benefit)`` — should we treat *everyone*? One
   cost-benefit call on the average effect.
2. ``result.decide_by_group()`` — should we treat each *pre-defined segment*?
   Requires you to choose the segmentation up front via ``effect_modifier``.
3. ``result.learn_policy(...)`` — learns the segmentation itself: given
   several candidate features, it finds the treatment rule that maximises
   net benefit, and reports an honest estimate of what that rule is worth.

The output is deliberately a *shallow decision tree* — e.g. "treat if
tenure = <2y" — because a rule a stakeholder can read, audit, and ship is
worth more than an opaque scoring model.

Policy learning is currently available on :class:`~formative.causal.RCTResult`:
with randomised treatment the machinery needs no propensity model, so there is
one less thing to get wrong.

How it works
------------

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
-------

A company runs a randomised coaching programme and measures employee
retention. Coaching costs $2.5k per participant and a unit of retention is
worth $1k. It helps recent hires a lot and everyone else not at all:

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
------------------

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
------------------------------------

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
-------------------------

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
---------------

- ``max_depth`` is the regulariser, and it is capped at 2 — deeper trees stop
  being auditable. Fit a depth-1 and a depth-2 policy and compare their
  ``value``: if the deeper tree isn't clearly worth more, ship the simpler one.
- The ``value`` describes the learning *procedure*, evaluated out-of-fold;
  the printed rule is re-learned on the full sample. This is what makes the
  estimate honest, at the cost of a slight mismatch between the rule you see
  and the folds behind the number.
- The treatment must be binary 0/1, and each arm needs at least 10 units for
  cross-fitting (in practice you want far more).

PolicyResult
------------

.. autoclass:: formative.causal.PolicyResult
   :members:

PolicyNode
----------

.. autoclass:: formative.causal.PolicyNode
   :members:

PolicyRefutationReport
----------------------

.. autoclass:: formative.causal.PolicyRefutationReport
   :members:
   :inherited-members:
