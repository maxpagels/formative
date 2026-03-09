Introduction
============

The game theory module provides decision rules for choosing between options under
uncertainty or in strategic situations. It is designed to be usable without a background
in game theory.

Every analysis follows two steps.

**1. Describe your options and their outcomes**

Declare each possible choice and what it yields under each scenario.

.. code-block:: python

   from formative.game import maximin

   outcomes = {
       "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
       "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
       "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
   }

**2. Apply a decision rule**

Choose a rule that matches how you want to reason about uncertainty.

.. code-block:: python

   result = maximin(outcomes).solve()
   print(result)

.. code-block:: text

   MaximinResult(
     stocks  worst case: -20
     bonds   worst case: +5  ← chosen
     cash    worst case: +2
   )
