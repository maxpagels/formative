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

Before any data is touched, you declare which variables cause which. This
makes your identification assumptions explicit and machine-readable.

.. code-block:: python

   from formative import DAG

   dag = DAG()
   dag.assume("ability").causes("education", "income")
   dag.assume("education").causes("income")

**2. Choose an estimator**

Pass the DAG to an estimator along with your treatment and outcome. The
estimator reads the DAG to determine which variables to control for (or
how to use an instrument), then fits the model. If the data cannot support
identification given the DAG, an error is raised before estimation runs.

.. code-block:: python

   from formative import OLSObservational

   result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
   print(result.summary())

**3. Refute**

Once you have a result, run statistical checks that probe whether its
assumptions hold in the data. Each check returns a clear pass or fail.

.. code-block:: python

   from formative import IV2SLS

   result = IV2SLS(dag, treatment="education", outcome="income", instrument="proximity").fit(df)
   report = result.refute(df)
   print(report.summary())

.. toctree::
   :maxdepth: 2
   :caption: Contents

   dag
   estimators
   refutations
