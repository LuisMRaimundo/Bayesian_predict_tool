# Audit response — scientific defects and corrections

This document tracks the external audit (ZIP ≡ `76b7cc4`) and the **re-audit** of commit `6f3c3dc`, with CI/docs follow-up at `e13b2d4` and later.

## Classification (current)

**An exploratory cross-corpus transfer tool with robust regression and Excel export — not a publication-grade hierarchical Bayesian extrapolation system.**

Defensible for: exploratory technique-conditioned EWSD profiles under **explicit** cross-corpus transport assumptions.  
Not defensible for: claiming how IOWA/ORCHIDEA would definitively have measured those techniques.

## Status of defects

| Previous finding | Status |
|---|---|
| Cross-collection pairs labelled as same-collection | **Fixed** |
| `require_same_collection=True` not enforced | **Fixed** |
| Separate source and ordinario corpus IDs missing | **Fixed** |
| Unknown dynamics assigned arbitrary ordinario dynamics | **Fixed** |
| Acoustic prior applied to every observation | **Largely fixed** (coefficient-level once) |
| Cross-validation winsorisation leakage | **Fixed** |
| Non-Bayesian M3 fallback presented as Bayesian | **Fixed** (hard-fail unless authorized; `is_bayesian=False`) |
| Corpus hierarchy in M3 | **Partial** — `(1\|corpus_id)` only if identifiable; absent when technique≡corpus |
| EWSD observation SE in Bayesian likelihood | **Not fixed** (`weighted=False`; needs custom PyMC) |
| Publication-grade M3 | **Not achieved** |
| `transport_sd` as corpus variance | **Clarified** as external proxy when technique≡corpus |
| CI / smoke failing under strict same-collection | **Fixed** (smoke + mode tests) |

## Re-audit execution notes (user violin bridge)

With corrected software on the Philharmonia-style bridge + IOWA/ORCHIDEA target:

- Strict same-collection mode **correctly rejects** the bridge.
- Cross-collection mode yields ~140 transport-prior log-ratios (100% transport).
- Unspecified-dynamic techniques (artificial harmonics, sul tasto) are **dynamic extrapolations**.
- Corrected blocked CV ≈ **18–20%** factor MAPE; M1 ranked best among M0–M2; M3 approx worst.

Recommended workflow for this material:

1. Enable cross-collection transport **explicitly**.  
2. Primary model: **M1**; baseline **M0**; secondary **M2**.  
3. Do **not** use M3 approximate.  
4. Describe every result as a **model-derived transport estimate**.  
5. Keep heuristic intervals; do **not** call them Bayesian credible intervals.

## Remaining scientific work

1. Same-collection / multi-session bridges so technique ≠ corpus.  
2. Custom PyMC likelihood \(r_i\sim t_\nu(\mu_i,\sqrt{\mathrm{SE}_i^2+\sigma^2})\).  
3. Outer validation when auto-selection + conformal calibration are both on.  
4. Re-audit after those land before publication claims.

See also: `LITERATURE_ALIGNMENT.md`, `requirements-bayes.txt`, `TECHNICAL_MANUAL.md`.
