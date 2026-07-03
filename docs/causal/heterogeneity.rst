Heterogeneous effects (CATE)
============================

An average treatment effect answers "does the intervention work?". In
practice you often need "*for whom* does it work?" — should a discount go to
new customers or loyal ones, does the training help juniors more than
seniors? These per-group effects are called **conditional average treatment
effects** (CATE).

``OLSObservational`` and ``RCT`` accept an ``effect_modifier``: a discrete,
pre-treatment column whose levels the effect is allowed to differ across.
Under the hood the fitted model becomes
``outcome ~ treatment * C(modifier) + controls``, and ``fit()`` returns a
result with per-group effects, a homogeneity test, and per-group decisions.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from formative.causal import DAG, OLSObservational

   rng = np.random.default_rng(42)
   n = 2_000
   segment = rng.integers(0, 3, size=n)
   ability = rng.normal(size=n)
   education = ability + rng.normal(size=n)
   # True effect differs by segment: 1.0, 2.0, 3.0
   income = (1 + segment) * education + 1.5 * ability + 0.5 * segment + rng.normal(size=n)
   df = pd.DataFrame({"segment": segment, "ability": ability,
                      "education": education, "income": income})

   dag = DAG()
   dag.assume("ability").causes("education", "income")
   dag.assume("segment").causes("income")
   dag.assume("education").causes("income")

   result = OLSObservational(
       dag, treatment="education", outcome="income", effect_modifier="segment"
   ).fit(df)
   print(result.summary())

The summary gains a per-group block:

.. code-block:: text

     Effect by segment  (homogeneity: F = 1322.98, p = 0.0000)
     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
       0 :     1.0296  [0.9673, 1.0919]  n=673
       1 :     2.0176  [1.9526, 2.0826]  n=660
       2 :     3.0412  [2.9775, 3.1048]  n=667

The **homogeneity test** asks whether the group differences are real or
noise: a small p-value rejects the hypothesis that the effect is the same in
every group. The headline ``result.effect`` is the sample-share-weighted
average of the group effects, so downstream tools (``decide()``, refutation
checks) keep working unchanged.

Deciding per group
------------------

``decide_by_group(cost, benefit)`` runs the cost-benefit analysis separately
for each modifier level, so different groups can receive different treatment
decisions:

.. code-block:: python

   decisions = result.decide_by_group(cost=2.5, benefit=1.0)
   for level, report in decisions.items():
       print(level, report.optimal, f"net benefit: {report.net_benefit:+.2f}")

.. code-block:: text

   0 don't treat net benefit: -1.47
   1 don't treat net benefit: -0.48
   2 treat net benefit: +0.54

Rules the modifier must satisfy
-------------------------------

The effect modifier must be a node in the DAG, and:

- **Not a descendant of the treatment.** Segmenting by a post-treatment
  variable (a mediator) manufactures artificial heterogeneity: the treated
  and untreated members of such a "segment" are not comparable populations.
  This raises a ``ValueError``.
- **An ancestor of the outcome.** A variable that modifies the effect on the
  outcome is a cause of it — assert the edge in the DAG.
- **Discrete, with at least 2 levels**, and the treatment must vary within
  every level (otherwise that group's effect is not estimable). Bin
  continuous moderators before use.

The DAG can only validate what you declare: if a modifier was in fact
measured *after* treatment but is not modelled as a descendant, no software
can catch it. Only segment by variables measured (or fixed) before treatment
was assigned.

Refutations
-----------

``result.refute(df)`` on a heterogeneous-effects result runs three checks:

- **Random common cause** — adding a pure-noise control must not shift the
  weighted average effect by more than one standard error.
- **Placebo modifier** — randomly permuting the modifier column must make
  the heterogeneity vanish (non-significant homogeneity test). Heterogeneity
  that survives having its modifier scrambled is an artifact.
- **Random modifier** — interacting the treatment with a pure-noise column
  instead must show no significant heterogeneity.

OLSCATEResult
-------------

.. autoclass:: formative.causal.OLSCATEResult
   :members:
   :inherited-members:

RCTCATEResult
-------------

.. autoclass:: formative.causal.RCTCATEResult
   :members:
   :inherited-members:

GroupEffect
-----------

.. autoclass:: formative.causal.GroupEffect
   :members:
