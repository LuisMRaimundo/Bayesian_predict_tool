# Methodology — String Technique Transfer

**Repository:** https://github.com/LuisMRaimundo/Bayesian_predict_tool  
**Local folder:** `C:\Users\lmr20\Desktop\Bayesian Tool`

## Scientific claim

This tool produces **model-derived synthetic estimates** of missing special-technique conditions by transporting technique effects estimated on a bridge corpus onto a target ordinario profile:

\[
Y_{c,t,d,p}=Y_{c,\mathrm{ordinario},d,p}\cdot\exp(\delta_{t,d,p})
\]

They must **not** be labelled as measured IOWA/ORCHIDEA (or other collection) observations unless the bridge is truly paired under the same recording conditions.

## Zenodo dynamics

For each instrument×collection, Zenodo ordinario workbooks expose three dynamics: **pp, mf, ff**.

Adequate bridge partners (default):

| Zenodo | Adequate bridge dynamics |
|---|---|
| pp | pp, p |
| mf | mf, mp |
| ff | ff, f |

Example: bridge sordina **forte** supports only Zenodo **ff**.

## Robust estimation pipeline

1. **Preflight** — verify ordinario+technique presence, bridge pairs, supported Zenodo dynamics.
2. **Bridge log-ratios** — pair by instrument/MIDI; nearest adequate dynamic only; winsorize; shrink toward acoustic priors.
3. **Model** — default **M2** regularized robust transfer (constant for thin designs; mild MIDI smooth otherwise).
4. **Predict** — clip to technique-plausible factor bounds; inflate SE for register/dynamic extrapolation; flag target outliers.
5. **Support split** — `Predictions_supported` vs `Predictions_all`.
6. **Blocked pitch CV** — leave contiguous MIDI blocks out; report MAE/RMSE on log-ratio and MAPE on factors.

## Acoustic priors (soft)

| Technique | Expected direction | Typical factor band |
|---|---|---|
| con sordino | decrease | 0.45–1.05 |
| sul ponticello | increase | 0.90–2.40 |
| sul tasto | decrease | 0.50–1.10 |
| natural/artificial harmonics | decrease | 0.25–1.05 |

## Recommended use

1. Run **Preflight**.
2. Keep **Strict dynamics** ON.
3. Prefer **M2** unless the bridge is rich (many techniques/dynamics) and Bayes extras work.
4. Publish/analyse only this triad (yellow-highlighted in the Excel workbook):
   1. **File** — your output `.xlsx`
   2. **Page** — **`Predictions_supported`**
   3. **Column** — **`y_pred`**
5. Report transport-prior fraction and blocked-CV errors with any released table.
6. Archive the run via `outputs/run_history/…/RUN_REPORT.html` (lists every uploaded Excel; charts fill after **Fit & predict**). See `INDEX.md` in that folder.

## Run history

Preflight and Fit & predict each write a timestamped folder under `outputs/run_history/`. The primary human artifact is **`RUN_REPORT.html`** (Markdown/JSON/CSV twins alongside). Preflight reports intentionally omit prediction charts and fitted-model fields; transfer reports include them when the pipeline completes.
