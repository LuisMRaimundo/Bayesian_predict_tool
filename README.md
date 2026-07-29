# String Technique Transfer

Local research tool for **instrument-agnostic** technique transfer of spectral-density metrics (primarily `EWSD_score_acoustic_balanced`) across violin, viola, cello, and double bass.

**Repository:** https://github.com/LuisMRaimundo/Bayesian_predict_tool  
**Local folder:** `C:\Users\lmr20\Desktop\Bayesian Tool`

**Scientific status:** engineering audit **closed**; exploratory **M1** workflow **operational**; scientific validation **open** until a genuine paired corpus exists ([PAIRED_CORPUS.md](PAIRED_CORPUS.md), [AUDIT_RESPONSE.md](AUDIT_RESPONSE.md)).

- Default: same-collection pairing; M1 model; heuristic intervals  
- M3: PyMC heteroscedastic Student-t (**uses `se_log_obs`**); hard-fails without Bayes stack; refuses transport-only bridges by default  
- Cross-folder Philharmonia bridges: `--allow-cross-collection` → all `transport_prior`

See [METHODOLOGY.md](METHODOLOGY.md) · [TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md) · [REPRODUCE.md](REPRODUCE.md) · [LITERATURE_ALIGNMENT.md](LITERATURE_ALIGNMENT.md) · [requirements-bayes.txt](requirements-bayes.txt)

## Run history (per run)

Every **Preflight** / **Fit & predict** writes a timestamped folder under `outputs/run_history/<YYYYMMDD_HHMMSS>_<kind>/`:

| Artifact | Role |
|---|---|
| **`RUN_REPORT.html`** | Primary illustrated report: **all uploaded Excel names**, config, operations, fit/CV, Chart.js graphs |
| `RUN_REPORT.md` | Markdown twin |
| `run_manifest.json` | Machine-readable full record |
| CSV snapshots | `predictions_supported.csv`, `bridge_log_ratios.csv`, `preflight.csv`, … |
| `INDEX.md` / `index.csv` | Catalog of all runs (includes `uploaded_excels` column) |

**What fills vs stays empty**

- **Preflight** fills inputs, uploaded workbook names, config, operations, preflight table. Charts / fitted backend / predictions stay empty (expected).
- **Fit & predict** additionally fills charts (support levels, factors vs MIDI, ordinario vs `y_pred`, medians by dynamic), fit summary, and blocked CV when enabled.

Browse the index at `outputs/run_history/INDEX.md`. Charts use the Chart.js CDN (need network on first open).

## Quick start (GUI)

```bat
cd /d "C:\Users\lmr20\Desktop\Bayesian Tool"
python run_gui.py
```

1. Add bridge files (ordinario + special techniques)  
2. Choose Zenodo ordinario workbook + source (**`MEDIA`** default = `Violin_Media` columns **M/N/O** = Media pp/mf/**ff**; or `ORCH` / `IOWA`)  
3. Click **Preflight** (history HTML lists every selected Excel; charts wait for Fit)  
4. Keep **Strict dynamics** ON; prefer model **M1** (exploratory default). For Philharmonia-style bridges with different folder collections, enable **cross-collection transport** (CLI `--allow-cross-collection`) — results are transport priors, not same-corpus pairs.  
5. **Fit & predict** → use this triad (also highlighted yellow in the Excel `README` sheet):

   1. **File:** your output workbook (e.g. `outputs\transfer_audit.xlsx`)  
   2. **Page:** **`Predictions_supported`**  
   3. **Column:** **`y_pred`**  ← values that mimic the special technique on IOWA/ORCHIDEA

6. Open the illustrated history: path printed in the GUI / CLI as **Run history HTML**

## CLI

```bat
python -m string_technique_transfer.cli --bridge data\violin_bridge_panel.csv --target "C:\Users\lmr20\Desktop\VIOLIN_Zenodo_collections_Arco_normal.xlsx" --zenodo-collection MEDIA --model M1_register_dynamic --allow-cross-collection --out outputs\transfer_audit.xlsx

python -m string_technique_transfer.cli --bridge ... --target ... --preflight-only
```

Cross-collection bridges require `--allow-cross-collection`. M3 hard-fails without a working Bayes stack unless `--allow-m3-approx` (still not Bayesian).

## Robustness features

- Zenodo **MEDIA** target (`Violin_Media` M/N/O) + ORCH/IOWA sheets  
- **Same-collection** bridge pairing by default; cross-collection pairs labelled `transport_prior` only if allowed  
- Unspecified technique dynamics kept as `unspecified` (not invented from ordinario)  
- Winsorize responses; acoustic prior applied **once** at model coefficients (not per row)  
- Model comparison defaults to **M0–M2**; blocked CV (winsor inside folds), calibration, holdout, sensitivity  
- M3: **PyMC heteroscedastic** Student-t (`se_log_obs` in scale); Bambi only if PyMC fails; approx only if `--allow-m3-approx`  
- Paired-corpus gates: see `PAIRED_CORPUS.md` (synthetic template under `data/`)  
- Excel: `Predictions_supported!y_pred` + validation sheets; intervals are **heuristic predictive**  
- Run history HTML under `outputs/run_history/` (uploaded Excel names + charts after Fit)

## Tests

```bat
pip install pytest
pytest -q
```

## Models

| ID | Use |
|---|---|
| `M1_register_dynamic` | **Default / primary exploratory** (best CV on corrected cross-corpus violin bridge) |
| `M0_global_factor` | Sensitivity baseline |
| `M2_midi_gam` | Secondary pitch-smoothing comparison |
| `M3_hierarchical_bayes` | PyMC heteroscedastic Student-t (`√(SE²+σ²)`); needs paired rows by default; hard-fails if Bayes missing |
