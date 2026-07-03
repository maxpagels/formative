Benchmarks
==========

You shouldn't trust a causal estimation library at face value.
This page takes famous studies where the "right answer" is already established, runs them through
formative, and shows the results side by side with the published figures. Every code block is
runnable as-is: the data is downloaded from its canonical public source, so you can reproduce
each number yourself.

Each benchmark has three ingredients: a dataset with a known ground truth, the formative code to
estimate the effect, and a scorecard comparing our numbers against the published ones. Each is
also available as a runnable Jupyter notebook in the `notebooks/ directory of the repository
<https://github.com/maxpagels/formative/tree/main/notebooks>`_.

The LaLonde benchmark: does job training raise earnings?
---------------------------------------------------------

The `National Supported Work Demonstration <https://en.wikipedia.org/wiki/National_Supported_Work_Demonstration>`_
(NSW) was a 1970s US programme that gave disadvantaged workers 9–18 months of subsidised work
experience. Crucially, admission was **randomised**, so the programme doubles as a randomised
controlled trial: we know the true effect of the training on 1978 earnings.

Robert LaLonde's famous 1986 paper *"Evaluating the Econometric Evaluations of Training Programs
with Experimental Data"* used this to run a devastating test. He threw away the experimental
control group, replaced it with survey respondents from the Panel Study of Income Dynamics (PSID)
and the Current Population Survey (CPS) — people who were never in the programme — and asked
whether standard econometric methods could recover the experimental answer from this
observational data. Mostly, they could not: naive comparisons were wildly wrong, and regression
adjustment didn't rescue them. Dehejia & Wahba (1999) then revisited the data and showed that
propensity score methods land close to the experimental benchmark.

This makes the NSW data the classic stress test for observational causal methods, because it lets
us check three things at once:

1. Does :class:`~formative.causal.RCT` reproduce the published experimental effect (**$1,794** in
   the Dehejia-Wahba sample)?
2. Does the naive comparison on observational data go as badly wrong as LaLonde reported
   (**−$15,205** against PSID controls)?
3. Does :class:`~formative.causal.PropensityScoreMatching` pull the estimate back to the
   neighbourhood of the experimental answer, as Dehejia & Wahba found?

Loading the data
^^^^^^^^^^^^^^^^

The Dehejia-Wahba sample is hosted on `Rajeev Dehejia's data page
<https://users.nber.org/~rdehejia/nswdata2.html>`_. Each file has the same ten columns: a
treatment indicator, pre-treatment characteristics (age, education, race, marital status, and
real earnings in 1974 and 1975), and the outcome — real earnings in 1978 (``re78``).

.. code-block:: python

    import pandas as pd

    BASE = "https://users.nber.org/~rdehejia/data"
    COLS = [
        "treat", "age", "education", "black", "hispanic",
        "married", "nodegree", "re74", "re75", "re78",
    ]

    def load(name):
        return pd.read_csv(f"{BASE}/{name}.txt", sep=r"\s+", header=None, names=COLS)

    treated = load("nswre74_treated")      # 185 NSW participants
    experimental_control = load("nswre74_control")  # 260 randomised controls
    psid = load("psid_controls")           # 2,490 PSID respondents (never in NSW)

    experimental = pd.concat([treated, experimental_control], ignore_index=True)
    observational = pd.concat([treated, psid], ignore_index=True)

Step 1: the experimental answer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because NSW admission was randomised, the DAG is as simple as it gets — treatment causes the
outcome, and nothing causes treatment:

.. code-block:: python

    from formative.causal import DAG, RCT

    rct_dag = DAG()
    rct_dag.assume("treat").causes("re78")

    result = RCT(rct_dag, treatment="treat", outcome="re78").fit(experimental)
    print(result.summary())

.. code-block:: text

    RCT Causal Effect: treat → re78
      Estimand: ATE (average treatment effect)
    ──────────────────────────────────────────────────
      ATE estimate         :  1794.3424

      Std. error           :   632.8534
      95% CI               : [550.5745, 3038.1103]
      p-value              :     0.0048
      N                    :        445

**$1,794.34** — the published experimental benchmark for this sample is $1,794 (Dehejia & Wahba
1999, Table 3). This is the ground truth the observational methods below will be judged against.

Step 2: the naive comparison goes badly wrong
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we discard the experimental controls and compare NSW participants against PSID respondents,
as LaLonde did. The comparison group is now nothing like the treated group — PSID respondents
are older, better educated, far more likely to be employed — so the covariates confound the
treatment/outcome relationship. The observational DAG declares this:

.. code-block:: python

    from formative.causal import OLSObservational

    COVARIATES = ["age", "education", "black", "hispanic",
                  "married", "nodegree", "re74", "re75"]

    obs_dag = DAG()
    for c in COVARIATES:
        obs_dag.assume(c).causes("treat", "re78")
    obs_dag.assume("treat").causes("re78")

    result = OLSObservational(obs_dag, treatment="treat", outcome="re78").fit(observational)
    print(result.summary())

.. code-block:: text

    OLS Causal Effect: treat → re78
    ──────────────────────────────────────────────────
      Adjusted estimate    :   751.9464  (controlling for: age, black, education, hispanic, married, nodegree, re74, re75)
      Unadjusted estimate  : -15204.7775  (no controls)
      Confounding bias     : -15956.7239

      Std. error           :   915.2572
      95% CI               : [-1042.7399, 2546.6327]
      p-value              :     0.4114
      N                    :       2675

This single summary reproduces LaLonde's core finding. The unadjusted estimate — the answer you
would get by simply comparing average earnings between the two groups — is **−$15,204.78**,
matching the −$15,205 raw gap in the published data. Taken at face value it says the training
programme *destroyed* fifteen thousand dollars of annual earnings; in reality it measures how
much poorer NSW participants were than average PSID respondents to begin with. Linear regression
adjustment moves the estimate to $752 — the right sign, but still less than half the experimental
benchmark and statistically indistinguishable from zero, echoing LaLonde's conclusion that
regression on a badly mismatched comparison group cannot be trusted. The adjusted-vs-unadjusted
comparison that every formative result carries exists precisely to make this kind of confounding
visible.

Step 3: matching recovers the benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dehejia & Wahba's insight was that most PSID respondents are so unlike NSW participants that a
linear model extrapolates across incomparable people. Propensity score matching instead compares
each treated person only with the most similar controls:

.. code-block:: python

    from formative.causal import PropensityScoreMatching

    result = PropensityScoreMatching(obs_dag, treatment="treat", outcome="re78").fit(observational)
    print(result.summary())

.. code-block:: text

    PSM Causal Effect: treat → re78
      Estimand: ATT (average treatment effect on the treated)
    ──────────────────────────────────────────────────────
      ATT estimate         :  2125.7131  (controlling for: age, black, education, hispanic, married, nodegree, re74, re75)
      Unadjusted estimate  : -15204.7775  (naive mean difference)
      Confounding bias     : -17330.4906

      Std. error           :  1079.2522  (bootstrap, N=500)
      95% CI               : [-943.4393, 3438.7593]  (bootstrap percentile)
      p-value              :     0.0489
      N                    :       2675

From the same observational data that produced −$15,205 naively, matching estimates **$2,125.71**
— within one standard error of the experimental $1,794. Repeating the exercise with the much
larger CPS comparison group (15,992 controls, naive gap −$8,497.52) gives an ATT of
**$2,074.02**: two very different comparison groups, matched estimates a mere $50 apart, both
bracketing the experimental truth. Dehejia & Wahba's own matching estimates were $1,691 (PSID)
and $1,582 (CPS); the residual spread between their numbers and ours reflects implementation
choices (they used stratification and different matching variants; formative uses 1-to-1
nearest-neighbour matching with replacement), and later work (Smith & Todd 2005) showed estimates
in this range are sensitive to exactly such choices. All of them are within one standard error of
the experimental answer.

Scorecard
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Quantity
     - Published
     - formative
     - Verdict
   * - Experimental ATE (RCT)
     - $1,794
     - $1,794.34
     - exact match
   * - Naive mean difference, PSID controls
     - −$15,205
     - −$15,204.78
     - exact match
   * - Naive mean difference, CPS controls
     - −$8,498
     - −$8,497.52
     - exact match
   * - PSM ATT, PSID controls
     - $1,691 (DW 1999)
     - $2,125.71
     - within 1 SE of experimental truth
   * - PSM ATT, CPS controls
     - $1,582 (DW 1999)
     - $2,074.02
     - within 1 SE of experimental truth

The pattern is exactly what forty years of literature says it should be: the experimental
estimate is reproduced to the dollar, the naive observational comparison is catastrophically
biased (and formative's unadjusted estimate quantifies that bias to the dollar), and propensity
score matching recovers an estimate statistically indistinguishable from the experimental truth.

References
^^^^^^^^^^

- LaLonde, R. (1986). "Evaluating the Econometric Evaluations of Training Programs with
  Experimental Data." *American Economic Review*, 76(4), 604–620.
- Dehejia, R. & Wahba, S. (1999). "Causal Effects in Nonexperimental Studies: Reevaluating the
  Evaluation of Training Programs." *Journal of the American Statistical Association*, 94(448),
  1053–1062.
- Smith, J. & Todd, P. (2005). "Does matching overcome LaLonde's critique of nonexperimental
  estimators?" *Journal of Econometrics*, 125(1–2), 305–353.
- Data: `Rajeev Dehejia's NSW data page <https://users.nber.org/~rdehejia/nswdata2.html>`_.

The Card–Krueger benchmark: does raising the minimum wage kill jobs?
--------------------------------------------------------------------

On 1 April 1992, New Jersey raised its minimum wage from $4.25 to $5.05 an hour. Neighbouring
Pennsylvania kept its minimum at $4.25. The textbook competitive-market prediction is
unambiguous: a binding wage floor should reduce employment. Card & Krueger (1994) tested it by
surveying 410 fast-food restaurants — Burger King, KFC, Roy Rogers, and Wendy's — on both sides
of the state border, once in February–March 1992 (before the rise) and again in November–December
1992 (after). Because the two states share a labour market and an economic climate, Pennsylvania
serves as the counterfactual for what would have happened to New Jersey employment without the
wage rise.

This is *the* canonical difference-in-differences study, and its result was famously
counterintuitive: employment in New Jersey did **not** fall relative to Pennsylvania — the
published DiD estimate is **+2.76** full-time-equivalent (FTE) jobs per store (Table 3, standard
error 1.36). The benchmark for :class:`~formative.causal.DiD` is therefore: feed it the public
survey data and check that it reproduces that coefficient.

Loading the data
^^^^^^^^^^^^^^^^

The dataset is distributed on `David Card's data page <https://davidcard.berkeley.edu/data_sets.html>`_
as a zip containing ``public.dat`` (one row per store, both survey waves side by side) and a
codebook. Following Card & Krueger, employment is measured in full-time equivalents: full-time
staff plus managers plus half the part-time staff.

.. code-block:: python

    import io
    import urllib.request
    import zipfile

    import pandas as pd

    URL = "https://davidcard.berkeley.edu/data_sets/njmin.zip"
    COLS = [
        "SHEET", "CHAIN", "CO_OWNED", "STATE", "SOUTHJ", "CENTRALJ", "NORTHJ",
        "PA1", "PA2", "SHORE", "NCALLS", "EMPFT", "EMPPT", "NMGRS", "WAGE_ST",
        "INCTIME", "FIRSTINC", "BONUS", "PCTAFF", "MEALS", "OPEN", "HRSOPEN",
        "PSODA", "PFRY", "PENTREE", "NREGS", "NREGS11", "TYPE2", "STATUS2",
        "DATE2", "NCALLS2", "EMPFT2", "EMPPT2", "NMGRS2", "WAGE_ST2", "INCTIME2",
        "FIRSTIN2", "SPECIAL2", "MEALS2", "OPEN2R", "HRSOPEN2", "PSODA2",
        "PFRY2", "PENTREE2", "NREGS2", "NREGS112",
    ]

    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())) as z:
        raw = pd.read_csv(z.open("public.dat"), sep=r"\s+", header=None,
                          names=COLS, na_values=".")

    # FTE employment = full-timers + managers + 0.5 × part-timers (CK's definition)
    raw["fte_before"] = raw["EMPFT"] + raw["NMGRS"] + 0.5 * raw["EMPPT"]
    raw["fte_after"] = raw["EMPFT2"] + raw["NMGRS2"] + 0.5 * raw["EMPPT2"]

Before estimating anything, we can check the raw ingredients against the paper. Card & Krueger's
Table 3 reports four cell means; formative's DiD will be built from exactly these:

.. code-block:: python

    for state, name in [(1, "NJ"), (0, "PA")]:
        sub = raw[raw.STATE == state]
        print(f"{name}: before {sub.fte_before.mean():.2f}, after {sub.fte_after.mean():.2f}")

.. code-block:: text

    NJ: before 20.44, after 21.03
    PA: before 23.33, after 21.17

All four match the published Table 3 to the last digit (NJ: 20.44 → 21.03; PA: 23.33 → 21.17).
Note the shape of the problem already visible here: New Jersey stores were *smaller* than
Pennsylvania stores before the wage rise, so a naive post-period comparison would be biased
against New Jersey — the exact baseline difference DiD exists to remove.

Running DiD
^^^^^^^^^^^

:class:`~formative.causal.DiD` expects long-format data — one row per store per period — with
binary group and time indicators:

.. code-block:: python

    from formative.causal import DAG, DiD

    long = pd.concat(
        [
            pd.DataFrame({"store": raw.SHEET, "nj": raw.STATE, "after": 0, "fte": raw.fte_before}),
            pd.DataFrame({"store": raw.SHEET, "nj": raw.STATE, "after": 1, "fte": raw.fte_after}),
        ],
        ignore_index=True,
    ).dropna(subset=["fte"])

    dag = DAG()
    dag.assume("nj").causes("fte")
    dag.assume("after").causes("fte")

    result = DiD(dag, group="nj", time="after", outcome="fte").fit(long)
    print(result.summary())

.. code-block:: text

    DiD Causal Effect: (nj × after) → fte
      Estimand: ATT (average treatment effect on the treated)
    ──────────────────────────────────────────────────────
      DiD estimate         :     2.7536
      Naive post-diff      :    -0.1382  (treated post − control post)
      Baseline bias removed:    -2.8918

      Std. error           :     1.6884
      95% CI               : [-0.5607, 6.0679]
      p-value              :     0.1033
      N                    :        794

The DiD estimate is **+2.75** FTE jobs — Card & Krueger's published figure is 2.76, an agreement
to within a hundredth of an FTE (rounding-level noise; note that the published two-decimal cell
means themselves difference to 2.75). The naive
post-period difference of −0.14 also matches the paper's reported NJ−PA gap of −0.14, and the
summary shows DiD stripping out the −2.89 baseline difference between the states. The sign of
the result is the famous finding: no evidence that the minimum wage rise reduced employment.

Restricting to the balanced panel — the 384 stores with employment measured in both waves —
reproduces the paper's balanced-sample row exactly:

.. code-block:: python

    balanced = raw.dropna(subset=["fte_before", "fte_after"]).SHEET
    result_bal = DiD(dag, group="nj", time="after", outcome="fte").fit(
        long[long.store.isin(balanced)]
    )
    print(f"balanced-sample DiD: {result_bal.effect:.4f}")

.. code-block:: text

    balanced-sample DiD: 2.7500

against a published 2.75 (Table 3, row 5, where the underlying mean changes are NJ +0.47 and
PA −2.28 — both of which this data reproduces exactly).

A note on the standard error: formative reports 1.69, computed from the pooled OLS interaction
model on store-periods. Card & Krueger's published 1.36 comes from a different (and for panel
data, sharper) calculation — the variance of the store-level *changes*, which nets out
persistent store-to-store size differences before computing uncertainty. The same data
reproduces their number too: the change-score formula on the balanced panel gives an SE of
1.3423 against their published 1.34. The point estimate is identical either way; the two SEs
answer slightly different questions about it, and formative's is the more conservative of the
two.

Scorecard
^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Quantity
     - Published
     - formative
     - Verdict
   * - Four Table 3 cell means (NJ/PA × before/after)
     - 20.44, 21.03, 23.33, 21.17
     - 20.44, 21.03, 23.33, 21.17
     - exact match
   * - Naive post-period NJ−PA difference
     - −0.14
     - −0.1382
     - exact match
   * - DiD estimate, all available stores
     - +2.76 (SE 1.36)
     - +2.7536 (SE 1.69)
     - match (see SE note)
   * - DiD estimate, balanced panel
     - +2.75 (SE 1.34)
     - +2.7500 (change-score SE 1.3423)
     - exact match

References
^^^^^^^^^^

- Card, D. & Krueger, A. (1994). "Minimum Wages and Employment: A Case Study of the Fast-Food
  Industry in New Jersey and Pennsylvania." *American Economic Review*, 84(4), 772–793.
- Data: `David Card's data page <https://davidcard.berkeley.edu/data_sets.html>`_ (``njmin.zip``).
