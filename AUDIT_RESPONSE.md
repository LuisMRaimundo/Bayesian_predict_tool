# Audit response — scientific defects and corrections

This document tracks the external audit verdict (ZIP ≡ GitHub `76b7cc4`) and the remediation status in this repository.

## Verdict accepted

The tool remains useful for **exploratory** transfer analysis and Excel export. It is **not** publication-ready as a hierarchical Bayesian extrapolation of IOWA/ORCHIDEA until the defects below are closed and re-audited on real multi-corpus bridges.

## Defect → correction map

| # | Audit finding | Status in code |
|---|---|---|
| 1 | Cross-collection pairs labelled `paired_same_collection` | **Fixed** — `bridge.py` enforces `require_same_collection` literally; cross-collection pairs only when allowed and labelled `transport_prior`; stores `special_corpus_id` + `ordinario_corpus_id` |
| 2 | M3 Bambi formula not hierarchical | **Partially fixed** — adds `(1\|corpus_id)` only when ≥1 technique spans multiple corpora; otherwise documents confounding and raises transport_sd floor |
| 3 | Observation SE unused in Bayes likelihood | **Partial** — SE retained; inverse-variance used in M2 WLS; full heteroskedastic Student-t still needs custom PyMC (documented, not silently claimed) |
| 4 | Silent non-Bayesian M3 fallback | **Fixed** — `allow_m3_approx_fallback=False` by default; hard `RuntimeError` unless authorized; package versions recorded when Bayes runs |
| 5 | Unspecified dynamics invented from ordinario | **Fixed** — `dynamic=unspecified`, `dynamic_support=unknown`; predictions → `extrapolated_dynamic` |
| 6 | Response-level acoustic prior dominates | **Fixed** — no per-row shrink in bridge; coefficient-level shrink once with weaker `prior_strength=1`; sensitivity toggles model prior |
| CV leakage | Winsorize before folds | **Fixed** — `blocked_cv` re-winsorizes inside each training fold |
| Intervals | Named as if Bayesian | **Fixed** — `interval_type=heuristic_predictive…` columns/diagnostics |
| Defaults | `require_same_collection=False` | **Fixed** — default `True` (GUI + config); CLI `--allow-cross-collection` to opt in |

## What users should do now

1. Prefer **same-collection bridges** (ordinario + technique from the same corpus folder).  
2. Treat cross-collection runs as **transport priors** (uncertainty inflated).  
3. Do **not** use M3 outputs unless Bayes extras install cleanly **and** the backend reports `bambi_pymc` with `is_bayesian=True`.  
4. After unspecified-dynamic fix, re-run model comparison before trusting M1 vs M2 ranks.  
5. Do **not** publish IOWA/ORCHIDEA extrapolations until a fresh audit confirms items 1–5 on your corpus.

## Remaining scientific work

- Custom PyMC likelihood \(r_i \sim t_\nu(\mu_i,\sqrt{\mathrm{SE}_i^2+\sigma^2})\).  
- Replicated multi-corpus bridges before estimating technique×corpus random effects.  
- Outer validation loop when auto model selection + conformal calibration are both on.  
- Optional CI job that runs a minimal real M3 sample (heavy; gated).

See also: `LITERATURE_ALIGNMENT.md`, `TECHNICAL_MANUAL.md` §1.6, `requirements-bayes.txt`.
