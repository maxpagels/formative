formative
=========

formative is a Python package for causal estimation and game theory, two intertwined fields of decision science. It is intended
as a practical tool with minimal jargon for those who want to apply these methods to real-world problems.

Install with pip:

.. code-block:: bash

   pip install formative-ds

**Causal estimation** (``formative.causal``) helps you estimate the effect of one variable
on another from observational or experimental data, whilst making your assumptions explicit
and testable. :doc:`Get started <causal/introduction>`.

**Game theory** (``formative.game``) provides decision rules for choosing between options
under uncertainty or in strategic situations. :doc:`Get started <game/introduction>`.

.. toctree::
   :maxdepth: 2
   :caption: Causal Estimation
   :hidden:

   causal/introduction
   causal/motivation
   causal/dag
   causal/estimands
   causal/wizard
   causal/estimators
   causal/refutations
   causal/decisions

.. toctree::
   :maxdepth: 2
   :caption: Game Theory
   :hidden:

   game/introduction
   game/game
