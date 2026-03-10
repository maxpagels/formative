Introduction
============

Game theory, much like causal estimation, is difficult to learn. The best way to learn is to practice.
formative includes a game module that aims to make using game theory in decision-making easier.

Every analysis follows two steps.

**1. Describe your options and their outcomes**

Declare each possible choice and what payoffs you estimate they will yield under each scenario.
Negative payoffs represent losses, positive payoffs represent gains.

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
