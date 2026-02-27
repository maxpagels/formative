formative
=========

formative is Python package for causal effect estimation.
It is designed to help practitioners estimate causal effects from observational data using a variety of methods, 
whilst guiding them through the process of model specification, estimation, and interpretation.

.. code-block:: python

   from formative import DAG, OLSObservational

   dag = DAG()
   dag.assume("ability").causes("education", "income")
   dag.assume("education").causes("income")

   result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
   print(result.summary())

.. toctree::
   :maxdepth: 2
   :caption: Contents

   dag
   estimators
