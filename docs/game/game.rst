Decision Rules
==============

All decision rules take the same input — a dict of ``{choice: {scenario: payoff}}`` —
and return a solver with a ``.solve()`` method. They differ only in the attitude towards
uncertainty they encode. Payoffs can be integers, or floats.

.. code-block:: python

   outcomes = {
       "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
       "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
       "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
   }

Maximin
-------

*Which choice has the best worst case?*

Pessimistic. Assumes the worst scenario will occur and picks the choice that hurts least.
Useful when downside protection matters more than upside.

.. code-block:: python

   from formative.game import maximin

   print(maximin(outcomes).solve())

.. code-block:: text

   MaximinResult(
     stocks  worst case: -20
     bonds   worst case: +5  ← chosen
     cash    worst case: +2
   )

Bonds win: their worst case (recession, +5) is better than stocks (−20) or cash (+2).

.. autoclass:: formative.game.Maximin
   :members:

.. autoclass:: formative.game.MaximinResult
   :members:

Maximax
-------

*Which choice has the best best case?*

Optimistic. Assumes the best scenario will occur and picks accordingly.
Appropriate when you can absorb losses and want to maximise upside.

.. code-block:: python

   from formative.game import maximax

   print(maximax(outcomes).solve())

.. code-block:: text

   MaximaxResult(
     stocks  best case: +30  ← chosen
     bonds   best case: +7
     cash    best case: +2
   )

Stocks win: their best case (growth, +30) is highest, even though they have the worst
downside.

.. autoclass:: formative.game.Maximax
   :members:

.. autoclass:: formative.game.MaximaxResult
   :members:

Minimax Regret
--------------

*Which choice do you regret the least in hindsight?*

Regret in a given scenario is the gap between what you received and the best you could
have received in that scenario. Minimax regret picks the choice where the worst-case
regret is smallest — a middle ground between maximin and maximax.

First, find the best available payoff in each scenario:
recession = 5 (bonds), stagnation = 5 (stocks or bonds), growth = 30 (stocks).

Then compute regret for every combination — how much you miss out on versus the best choice:

.. list-table:: Regret table
   :header-rows: 1
   :stub-columns: 1

   * -
     - recession
     - stagnation
     - growth
     - max regret
   * - stocks
     - 25
     - 0
     - 0
     - 25
   * - bonds
     - 0
     - 0
     - 23
     - **23** ← lowest
   * - cash
     - 3
     - 3
     - 28
     - 28

.. code-block:: python

   from formative.game import minimax_regret

   result = minimax_regret(outcomes).solve()
   print(result)

.. code-block:: text

   MinimaxRegretResult(
     stocks  max regret: +25
     bonds   max regret: +23  ← chosen
     cash    max regret: +28
   )

Stocks score zero regret in growth (they are the best there) but 25 in recession. Bonds
score zero in recession and stagnation, but 23 in growth where stocks outperform them.
Cash never comes close to the best option in any scenario, giving it the highest max
regret of 28. Bonds minimise the worst-case regret.

.. autoclass:: formative.game.MinimaxRegret
   :members:

.. autoclass:: formative.game.MinimaxRegretResult
   :members:

Hurwicz Criterion
-----------------

*Which choice scores best on a blend of optimism and pessimism?*

The Hurwicz criterion scores each choice as:

.. math::

   \text{score} = \alpha \cdot \text{best case} + (1 - \alpha) \cdot \text{worst case}

The parameter ``alpha`` controls how optimistic you are. At ``alpha=0`` the rule is
identical to maximin (pure pessimism); at ``alpha=1`` it is identical to maximax (pure
optimism). Values in between express a considered attitude towards risk.

With ``alpha=0.5`` (equal weight to best and worst case):

- stocks: 0.5 × 30 + 0.5 × (−20) = **+5**
- bonds:  0.5 × 7  + 0.5 × 5     = **+6** ← chosen
- cash:   0.5 × 2  + 0.5 × 2     = **+2**

.. code-block:: python

   from formative.game import hurwicz

   print(hurwicz(outcomes, alpha=0.5).solve())

.. code-block:: text

   HurwiczResult(alpha=0.5,
     stocks  score: +5
     bonds   score: +6  ← chosen
     cash    score: +2
   )

At ``alpha=0.8`` the optimistic weight dominates and stocks pull ahead:

.. code-block:: python

   print(hurwicz(outcomes, alpha=0.8).solve())

.. code-block:: text

   HurwiczResult(alpha=0.8,
     stocks  score: +20  ← chosen
     bonds   score: +6.6
     cash    score: +2
   )

.. autoclass:: formative.game.Hurwicz
   :members:

.. autoclass:: formative.game.HurwiczResult
   :members:

Laplace Criterion
-----------------

*Which choice has the best average payoff?*

With no information about which scenario is more likely, treat them all as equally
probable (Laplace's principle of indifference) and pick the choice with the highest
average payoff.

- stocks: (−20 + 5 + 30) ÷ 3 = **+5.0**
- bonds:  (5 + 5 + 7) ÷ 3    = **+5.67** ← chosen
- cash:   (2 + 2 + 2) ÷ 3    = **+2.0**

.. code-block:: python

   from formative.game import laplace

   print(laplace(outcomes).solve())

.. code-block:: text

   LaplaceResult(
     stocks  average: +5
     bonds   average: +5.667  ← chosen
     cash    average: +2
   )

.. autoclass:: formative.game.Laplace
   :members:

.. autoclass:: formative.game.LaplaceResult
   :members:
