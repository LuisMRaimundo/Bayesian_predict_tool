# Audit response — scientific defects and corrections

Tracks the external audit (`76b7cc4`), re-audit (`6f3c3dc` / `e13b2d4`), and the
**scientific advance** commits (heteroscedastic PyMC + paired-corpus pathway).

## Classification (current)

| Layer | Status |
|---|---|
| **Engineering audit** | **Closed** |
| **Exploratory M1 workflow** | **Operational** |
| **Scientific validation** | **Open** until a genuine paired corpus exists |

Still: *exploratory cross-corpus transfer tool* for Philharmonia-style bridges.  
Not yet: *publication-grade reconstruction of IOWA/ORCHIDEA measurements*.

Toward 95–100 scientific readiness requires **real** same-chain paired recordings
(see `PAIRED_CORPUS.md`). Code alone cannot invent that identification.

## Defect status

| Finding | Status |
|---|---|
| Cross-collection mislabelled as same-collection | **Fixed** |
| Dual corpus IDs | **Fixed** |
| Unspecified dynamics invented | **Fixed** |
| Response-level acoustic prior | **Fixed** (coefficient-level) |
| CV winsor leakage | **Fixed** |
| Silent non-Bayesian M3 | **Fixed** |
| CI / smoke vs strict default | **Fixed** |
| EWSD SE unused in Bayes likelihood | **Fixed in code** — PyMC `StudentT(ν, μ, √(SE²+σ²))` (`models/m3_pymc.py`) |
| Corpus hierarchy when technique≡corpus | **Correctly refused** |
| Genuine paired experimental corpus | **Still required** (user data) |

## Scientific advances in code

1. **Heteroscedastic M3 (PyMC)**  
   Preferred M3 backend: `pymc_heteroscedastic_student_t` with
   `observation_se_in_likelihood=True`. Bambi remains a secondary fallback
   (homoscedastic). Approx remains opt-in only.

2. **Paired-corpus pathway**  
   - `paired_corpus.assess_paired_corpus`  
   - Preflight / pipeline diagnostics (`paired_corpus_tier`)  
   - `require_paired_corpus_for_m3=True` by default  
   - Guide: `PAIRED_CORPUS.md`  
   - Synthetic template: `data/paired_corpus_synthetic.csv` (CI only — not real acoustics)

## Recommended use (dissertation / research)

| Bridge type | Model | Claim language |
|---|---|---|
| Transport-only (current Philharmonia folders) | **M1** (+ M0/M2 sensitivity) | Model-derived **transport** estimates |
| Same-collection paired (new recordings) | M1 / M2 / **M3 PyMC** | Still synthetic, but technique not confounded with corpus |
| M3 approx | **Do not use** | Non-Bayesian |

## Remaining for true 95–100 scientific score

1. Record ≥1 genuine paired violin session (ordinario + techniques, annotated dynamics).  
2. Re-run blocked CV / holdout on that corpus.  
3. Optional: leave-one-corpus-out once ≥2 paired corpora exist.  
4. External re-audit of IOWA/ORCHIDEA extrapolations after (1)–(3).
