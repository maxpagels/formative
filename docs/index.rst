formative
=========

formative is Python package for causal effect estimation.
It is designed to help practitioners estimate causal effects from observational data using a variety of methods,
whilst guiding them through the process of model specification, estimation, and interpretation.

Causal estimation is full of statistical jargon, which hinders adoption. formative's goal is to make causal
estimation more accessible, It is this author's opinion that attempting to do causal estimation is better than
simply comparing averages or correlations.

Install formative with pip:

.. code-block:: bash

   pip install formative-ds

How it works
------------

Every analysis in formative follows three steps:

**1. Encode your causal assumptions as a DAG**

Before any data is touched, you declare which variables are assumed to cause which.
This makes your identification assumptions explicit and machine-readable. formative
uses the DAG to determine which variables to control for (or how to use an instrument),
and to check whether the data can support identification given those assumptions. Remember,
the DAG is not a data model, but your assumptions about the data generating process. A partial
DAG is better than no DAG.

.. code-block:: python

   from formative import DAG

   dag = DAG()
   dag.assume("proximity").causes("education")
   dag.assume("ability").causes("education", "income")
   dag.assume("education").causes("income")

**2. Choose an estimator**

Pass the DAG to an estimator along with your treatment and outcome. The
estimator reads the DAG to determine which variables to control for (or
how to use an instrument), then fits the model. If the data cannot support
identification given the DAG, an error is raised before estimation runs.

Choosing the right estimator for your causal question is crucial;
not all methods are possible given your problem.
See https://getformative.dev for an online wizard to help you choose.

.. code-block:: python

   from formative import IV2SLS

   result = IV2SLS(
       dag,
       treatment="education",
       outcome="income",
       instrument="proximity"
   ).fit(df)

After you have obtained the result object, you can print a summary of the
estimate and its assumptions. The assumptions are the conditions that should
hold for the estimate to be valid. They are marked as testable or untestable
depending on whether formative can check them. Most assumptions in causal inference
are untestable by nature, and are things you must argue for based on domain knowledge
and theory. That is what makes causal inference an exciting but challenging endeavour.

.. code-block:: python

   print(result.summary())

.. code-block:: text

   IV (2SLS) Causal Effect: education → income
     Instrument: proximity
   ──────────────────────────────────────────────────
     IV estimate          :     1.9643
     Unadjusted estimate  :     2.2598  (no controls)
     Confounding bias     :    +0.2955

     Std. error           :     0.0377
     95% CI               : [1.8905, 2.0381]
     p-value              :     0.0000

     Assumptions
     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
     [  testable  ]  Relevance: the instrument strongly affects treatment
     [ untestable ]  Exclusion restriction: instrument only affects outcome through treatment
     [ untestable ]  Independence: instrument is uncorrelated with unobserved confounders
     [ untestable ]  Monotonicity: instrument affects treatment in same direction for everyone
     [ untestable ]  Stable Unit Treatment Value Assumption (SUTVA)

It is not uncommon to have challenges communicating the need for proper causal estimation and associated
assumptions to non-technical stakeholders. ``executive_summary()`` gives you a more plain-english
summary of the result, with as little technical language as possible. This is useful for communicating
results to non-technical audiences.

**3. Refute**

Once you have a result, run statistical checks that probe whether its
assumptions hold in the data. Each check returns a clear pass or fail
but remember, no set of tests can guarantee validity. Use them as diagnostics,
not proof. Causal inference is a judgment call, not a mathematical certainty.

The most common check is deliberately introducing a random common cause (a "placebo" variable)
and seeing if the estimate changes. The full list of refutations depends on the chosen estimator.

.. code-block:: python

   report = result.refute(df)
   print(report.summary())

.. code-block:: text

   IV Refutation Report: education → income
     Instrument: proximity
   ──────────────────────────────────────────────────
     [PASS]  First-stage F-statistic: F = 911.22  (threshold: F ≥ 10)
     [PASS]  Random common cause: estimate shifted by 0.0001  (≤ 1 SE = 0.0377)

     All checks passed.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   motivation
   dag
   estimands
   wizard
   estimators
   refutations
   decisions
