# Causal Impact & Heterogeneous Response Analysis

A causal inference portfolio project built on a single genuinely randomized dataset. It first proves its own estimation pipeline works, by artificially confounding a subsample and checking whether propensity-based methods recover the known truth, then applies that validated pipeline to the full randomized data to estimate the average treatment effect and, more importantly, how that effect varies across user segments.

Runs entirely on a 16GB, no-GPU laptop using free-tier tools throughout.

---

## The narrative

1. **Prove the method works.** Artificially confound a subsample of the randomized data and check whether naive OLS, propensity score matching, IPW, and AIPW recover the known ground-truth effect across a range of confounding severities.
2. **Justify the outcome variable.** A minimum-detectable-effect calculation, not a default, decides whether `visit` or `conversion` is usable for segment-level analysis.
3. **Find who responds differently.** Estimate a CATE model on the clean randomized data, evaluate it honestly (never assume it's correct just because it ran), and segment users to see where the treatment effect actually differs.
4. **Check if that finding is statistically defensible.** Correct for testing many segments at once, and confirm each segment is even large enough to detect the effect being claimed.
5. **Stress-test the result.** Rosenbaum bounds quantify how much unmeasured confounding would be needed to overturn the matched-pairs conclusion.

Every method here serves that narrative. Nothing was added just to check a skill-list box, see [Explicitly Out of Scope](#explicitly-out-of-scope) for what was deliberately left out, and why.

---

## Dashboard Preview

Screenshots from `streamlit run dashboard/app.py`, one per pipeline section.

<p align="center">
  <img src="assets/01_validation.png" width="800" alt="Validation tab: bias-severity curve and covariate balance diagnostics">
  <br>
  <sub><b>Section 1 — Validation.</b> Bias-severity curve across confounding severities against the ground-truth ATE, plus post-matching covariate balance.</sub>
</p>

<br>

<p align="center">
  <img src="assets/05_outcome_justification.png" width="800" alt="Outcome Justification tab: MDE comparison between visit and conversion">
  <br>
  <sub><b>Section 1.5 — Outcome Justification.</b> Live MDE calculation showing why <code>visit</code>, not <code>conversion</code>, is used for segment-level work.</sub>
</p>

<br>

<p align="center">
  <img src="assets/02_heterogeneity.png" width="800" alt="Heterogeneity tab: segment-level CATE breakdown and Qini curve">
  <br>
  <sub><b>Section 2 — Heterogeneity.</b> Per-segment treatment effects with bootstrap CIs, and the Qini curve evaluating CATE model quality.</sub>
</p>

<br>

<p align="center">
  <img src="assets/03_statistical_rigor.png" width="800" alt="Statistical Rigor tab: Benjamini-Hochberg correction and per-segment power analysis">
  <br>
  <sub><b>Section 3 — Statistical Rigor.</b> Benjamini-Hochberg correction across segments, and per-segment power analysis flagging underpowered clusters.</sub>
</p>

<br>

<p align="center">
  <img src="assets/04_sensitivity.png" width="800" alt="Sensitivity tab: Rosenbaum bounds and LLM diagnostic critique">
  <br>
  <sub><b>Section 4 — Sensitivity.</b> Rosenbaum bounds on the PSM matched pairs, critical Gamma, and the bounded LLM diagnostic critique.</sub>
</p>

---

## Dataset

**[Criteo Uplift Modeling Dataset, v2.1](https://huggingface.co/datasets/criteo/criteo-uplift)** (Diemert et al., 2018), ~13.9M rows from a real, randomized ad-exposure experiment in a live advertising system. v2.1 adds an `exposure` column on top of `treatment`. No registration or approval process required.

| Column | Type | Description |
|---|---|---|
| `f0`–`f11` | float | 12 anonymized dense covariates |
| `treatment` | binary | 1 = treated, 0 = control |
| `exposure` | binary | effective exposure flag (v2.1 addition) |
| `visit` | binary | base rate ≈ 4–5% |
| `conversion` | binary | base rate ≈ 0.3% |

Loaded via `datasets.load_dataset("criteo/criteo-uplift")` (Hugging Face), with `sklift.datasets.fetch_criteo(target_col='all', treatment_col='all')` as an automatic fallback if Hugging Face is unreachable. An integrity check runs immediately after load, row count, column names, and treatment/control split ratio are all checked against documented values, and the load **fails loudly** on drift rather than silently continuing.

### Why one dataset, not two

An earlier version of this project used a second, older dataset (LaLonde, 1986) purely to provide a ground-truth check before trusting the pipeline on data without one. That role turned out to be redundant: Criteo is already randomized, so it already provides its own ground truth. Validation instead comes from artificially confounding a subsample of Criteo itself and checking recovery against the same dataset's true randomized answer.

### Why `visit`, not `conversion`

Before any segment-level work begins, a minimum detectable effect (MDE) calculation (`statsmodels.stats.power.NormalIndPower`) checks what effect size each outcome can actually detect at the planned subsample size:

| Outcome | Baseline rate | Relative MDE at n=100k/arm |
|---|---|---|
| `visit` | ~4.5% | ~5.9% |
| `conversion` | ~0.3% | ~24.1% |

`conversion`'s rarity means it can't support reliable per-segment estimation at CPU-feasible sample sizes. Consequence: **`visit` is the outcome for Sections 2 and 3.** `conversion` is used only for the full-dataset ground-truth ATE in Section 1, where n is large enough to be meaningful, and is explicitly excluded from segment-level work, this table is why, not an afterthought.

---

## Pipeline

### Section 1, Validation via self-induced confounding

The ground-truth ATE is computed directly from the full randomized dataset (analytic Wald CI, bootstrapping ~13.9M rows would cost real compute for no statistical gain at that sample size).

Confounding is induced with a parameterized retention rule, not by dropping units on an outcome-correlated covariate alone (which doesn't reliably bias a treatment effect estimate in already-randomized data):

```
retention_probability = sigmoid(g0 + g1·X + g2·X·T)
```

`X` is selected by correlation with the outcome, not arbitrarily, an outcome-irrelevant `X` would let `g2` grow indefinitely without ever biasing the naive estimate, since the whole point of the interaction term is that it only creates real confounding when `X` actually predicts the outcome.

A **calibration loop** increases `g2` until a two-part validation gate passes: (a) `corr(X, T)` is significantly nonzero in the retained sample, and (b) the naive difference-in-means estimate falls outside the ground-truth CI. The loop is capped at `max_iters` and reports `converged: False` explicitly if the gate never trips, rather than looping forever or silently accepting an uncalibrated severity.

The full estimator comparison, naive OLS, propensity score matching (with balance diagnostics), IPW, and AIPW, runs across four severities (none / mild / moderate / strong), producing the project's central diagnostic: a **bias-severity curve** showing naive estimation break down while matching/weighting stay comparatively robust.

### Section 2, Heterogeneity

A **T-learner** is the primary CATE method, simpler and more defensible to explain than a causal forest, which is documented as a future extension rather than built (see below). Base classifiers are calibrated (`CalibratedClassifierCV`), since `visit`'s low base rate makes uncalibrated probability estimates noisy in exactly the way that gets amplified by taking a difference of two of them.

The T-learner's output is never assumed correct just because it ran, a **Qini coefficient** against a held-out split quantifies whether it actually ranks units by uplift better than random targeting.

Users are segmented via unsupervised clustering on pre-treatment covariates (Criteo's covariates are anonymized floats, not literal recency/frequency/value fields, so clustering is the natural default; a business-style quantile split is available as a simpler alternative). Per-segment treatment effects are estimated with bootstrap confidence intervals.

### Section 3, Statistical rigor

Testing many segments at once inflates the false-positive rate, so **Benjamini-Hochberg correction** is applied before any segment is called significant. A **per-segment power analysis** separately checks whether each segment is even large enough to detect the effect size being claimed, a common real-world mistake this project deliberately surfaces rather than glosses over.

### Section 4, Sensitivity analysis

**Rosenbaum bounds** run specifically on the PSM matched-pairs output from Section 1, at the calibrated confounding severity, not on IPW/AIPW estimates, since Rosenbaum bounds are defined in terms of matched pairs and there's no equivalent structure for a weighting-based estimator. The output is the critical Gamma: how strong an unmeasured confounder would need to be, in odds-ratio terms, to overturn the matched-pairs conclusion.

### The one LLM step

A single diagnostic critique reads the balance table, overlap diagnostics, and Rosenbaum sensitivity output, and produces a short plain-language flag of likely assumption violations for a non-technical stakeholder. **Groq is the primary provider, NVIDIA NIM the fallback** (both free tier, both serving `openai/gpt-oss-120b`), with a deterministic rule-based fallback if neither is reachable. This is a small, clearly bounded diagnostic utility, it does not generate narrative reports, does not summarize the project, does not write this README, and is not used anywhere in Sections 2 or 3. Nothing else in this project calls an LLM.

---

## Explicitly out of scope

Documented here as deliberate decisions, not gaps in knowledge:

- **Causal forest / X-learner**, a natural extension of the T-learner work, not built, to protect feasibility on a 16GB laptop.
- **Recommendation systems**, a different problem family (ranking/retrieval), doesn't belong here.
- **Deep learning**, not needed for this problem, and against the hard constraints below.
- **MLflow/Dagshub experiment tracking**, that's a different project's territory; duplicating it here would dilute both.
- **A second dataset for validation**, addressed above; Criteo's own randomization made it redundant.
- **`conversion` as a segment-level outcome**, addressed above via the MDE calculation.
- **Difference-in-differences**, Criteo's rows are independent single-exposure-window observations with no pre/post structure, so there's no usable time dimension for DiD.
- **Any LLM usage beyond the one diagnostic critique step** described above.

## Hard constraints

- 16GB RAM, no GPU.
- Free-tier cloud services only.
- Every method included serves the causal narrative, nothing is here just to check a skill-list box.

---

## Project structure

```
causal-uplift-project/
├── data/                          # local dataset cache + generated artifacts (gitignored)
│
├── src/
│   ├── utils/
│   │   ├── data_loader.py         # HF load + sklift fallback + integrity check
│   │   ├── power_analysis.py      # MDE calc, power curves, per-segment power
│   │   ├── bootstrap.py           # stratified bootstrap CI + analytic large-n CI
│   │   ├── db.py                  # Supabase run-logging + local SQLite fallback
│   │   └── logging_config.py      # Logfire-backed structured logging + console fallback
│   │
│   ├── validation/                # Section 1
│   │   ├── confounding.py         # retention rule, calibration loop, dose-response grid
│   │   ├── estimators.py          # naive OLS, PSM, IPW, AIPW, common-support trimming
│   │   └── diagnostics.py         # balance (SMD, love plot), overlap diagnostics
│   │
│   ├── heterogeneity/             # Section 2 + 3
│   │   ├── cate.py                # T-learner (calibrated base classifiers)
│   │   ├── segmentation.py        # clustering / quantile splits, per-segment effects
│   │   └── evaluation.py          # Qini coefficient, Benjamini-Hochberg correction
│   │
│   ├── sensitivity/               # Section 4
│   │   └── rosenbaum.py           # Rosenbaum bounds on PSM matched pairs
│   │
│   └── llm_critique/
│       └── critique.py            # the one bounded LLM diagnostic step
│
├── notebooks/
│   ├── 01_validation.ipynb        # Section 1
│   ├── 02_heterogeneity.ipynb     # Section 2 and 3
│   └── 03_sensitivity.ipynb       # Section 4
│
├── dashboard/app.py               # Streamlit dashboard (reads db.py + data/processed/ artifacts)
│
├── tests/
│
├── .env.example
├──.gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <this-repo>
cd causal-uplift-project
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env            # then fill in whichever keys you're using (all optional)
```

> **Known install gotcha (Debian/Ubuntu):** installing `supabase` can fail to upgrade a
> system-installed `PyJWT` package ("Cannot uninstall PyJWT ... RECORD file not found"). If you
> hit this, run `pip install supabase --ignore-installed PyJWT`.

### Environment variables (all optional, everything has a local fallback)

| Variable | Used by | If unset |
|---|---|---|
| `GROQ_API_KEY` | LLM critique (primary) | Falls back to NVIDIA NIM, then to a rule-based critique |
| `NVIDIA_NIM_API_KEY` | LLM critique (fallback) | Falls back to a rule-based critique |
| `SUPABASE_URL` / `SUPABASE_KEY` | Run logging | Falls back to a local SQLite file at `data/local_runs.db` |
| `LOGFIRE_TOKEN` | Structured logging | Falls back to plain console logging |

## Running the pipeline

```bash
jupyter notebook
```

Run the three notebooks in order, each is a thin wrapper that calls into `src/`, so the actual logic lives in one tested place, not duplicated across notebook and dashboard:

1. `01_validation.ipynb`, Sections 1 & 1.5. Saves `data/processed/ground_truth.json`, `data/processed/balance_table.csv`, and `data/interim/confounded_strong_with_propensity.parquet` (consumed by notebook 3), and logs every estimator run to the database.
2. `02_heterogeneity.ipynb`, Sections 2 & 3. Saves `data/processed/qini_curve.json`, `data/processed/segment_effects.csv`, `data/processed/segment_power.csv`.
3. `03_sensitivity.ipynb`, Section 4. Loads notebook 1's saved artifact, saves `data/processed/rosenbaum_bounds.csv` and `data/processed/critique.json`.

Then view everything together:

```bash
streamlit run dashboard/app.py
```

The dashboard is a visualization layer, not a recomputation engine, it reads the logged runs and saved artifacts above. If a section's artifact doesn't exist yet, it shows which notebook to run, rather than crashing.

## Running the tests

```bash
pytest tests/ -v
```

25 tests across `test_confounding.py`, `test_estimators.py`, and `test_power_analysis.py`, covering calibration convergence/non-convergence, estimator correctness on known synthetic data, and MDE/power calculation correctness.

---

## Tech stack

| Layer | Choice |
|---|---|
| Compute | Local CPU only, subsampled for CATE/estimator work |
| Core libraries | `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `scipy`, `scikit-uplift` |
| Dashboard | Streamlit |
| Database | Supabase (primary), local SQLite (automatic fallback) |
| Logging | Logfire (structured), console (automatic fallback) |
| LLM | Groq (primary) → NVIDIA NIM (fallback), both serving `openai/gpt-oss-120b`, free tier |

---