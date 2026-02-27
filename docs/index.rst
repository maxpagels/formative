formative
=========

formative is Python package for causal effect estimation.
It is designed to help practitioners estimate causal effects from observational data using a variety of methods,
whilst guiding them through the process of model specification, estimation, and interpretation.

Causal estimation is full of statistical jargon, which hinders adoption. formative's goal is to make causal
estimation more accessible, because any attempt to understand the world causally is better than none at all.

How it works
------------

Every analysis in formative follows three steps:

**1. Encode your causal assumptions as a DAG**

Before any data is touched, you declare which variables are assumed to cause which.
This makes your identification assumptions explicit and machine-readable.

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

.. code-block:: python

   from formative import IV2SLS

   result = IV2SLS(
       dag, treatment="education", outcome="income", instrument="proximity"
   ).fit(df)
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

     Validity relies on the instrument being relevant (correlated
     with treatment) and satisfying the exclusion restriction
     (affecting outcome only through treatment). The DAG checks
     structural validity but cannot verify these empirically.

**3. Refute**

Once you have a result, run statistical checks that probe whether its
assumptions hold in the data. Each check returns a clear pass or fail
but remember, no set of tests can guarantee validity. Use them as diagnostics,
not proof. Causal inference is a judgment call, not a mathematical certainty.

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

   dag
   estimators
   refutations
