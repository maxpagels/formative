Game Theory
===========

formative includes a game theory module for analysing strategic decisions.
It is designed to be usable without a background in game theory.

Maximin
-------

The **maximin** rule answers a simple question: *which choice has the best worst case?*

For each option, identify the worst possible outcome across all scenarios. Then pick the
option whose worst case is the least bad. This is useful when you are uncertain about the
future and want a strategy that is robust regardless of what happens.

.. code-block:: python

   from formative.game import maximin

   result = maximin({
       "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
       "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
       "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
   }).solve()

   print(result)

.. code-block:: text

   MaximinResult(
     stocks  worst case: -20
     bonds   worst case: +5  ← chosen
     cash    worst case: +2
   )

Stocks offer the highest upside in a growth economy but expose you to a loss of 20 in a
recession. Bonds guarantee at least 5 in every scenario — the best guaranteed floor of the
three options. Cash is safer than stocks in a downturn but still dominated by bonds on the
worst case.

``maximin`` takes any dict of ``{choice: {scenario: payoff}}``.
