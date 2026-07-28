# String Technique Transfer

Local research tool for **instrument-agnostic** technique transfer of spectral-density metrics (primarily `EWSD_score_acoustic_balanced`) across violin, viola, cello, and double bass.

**Repository:** https://github.com/LuisMRaimundo/Bayesian_predict_tool  
**Local folder:** `C:\Users\lmr20\Desktop\Bayesian Tool`

See [METHODOLOGY.md](METHODOLOGY.md) for the scientific protocol.  
Full math / algorithms / code-line map + tutorial: **[TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md)** (open in [StackEdit](https://stackedit.io); LaTeX twin in `docs/TECHNICAL_MANUAL.tex`).  
Reproduce: [REPRODUCE.md](REPRODUCE.md) · Literature priors: [LITERATURE_ALIGNMENT.md](LITERATURE_ALIGNMENT.md)

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
4. Keep **Strict dynamics** ON; model **M2**  
5. **Fit & predict** → use this triad (also highlighted yellow in the Excel `README` sheet):

   1. **File:** your output workbook (e.g. `outputs\transfer_audit.xlsx`)  
   2. **Page:** **`Predictions_supported`**  
   3. **Column:** **`y_pred`**  ← values that mimic the special technique on IOWA/ORCHIDEA

6. Open the illustrated history: path printed in the GUI / CLI as **Run history HTML**

## CLI

```bat
python -m string_technique_transfer.cli --bridge data\violin_bridge_panel.csv --target "C:\Users\lmr20\Desktop\VIOLIN_Zenodo_collections_Arco_normal.xlsx" --zenodo-collection MEDIA --model M2_midi_gam --out outputs\transfer_audit.xlsx

python -m string_technique_transfer.cli --bridge ... --target ... --preflight-only
```

## Robustness features

- Zenodo **MEDIA** target (`Violin_Media` M/N/O) + ORCH/IOWA sheets  
- Strict dynamics map; winsorize + soft acoustic shrink/clip (literature-aligned, not activated EWSD laws)  
- M0–M3 comparison, blocked CV, residual/conformal **calibration**, holdout pack, sensitivity grid  
- M3: Bambi posterior mean when available, else hierarchical approx with corpus pooling  
- Excel: `Predictions_supported!y_pred` + `Model_comparison` / `Calibration` / `Holdout_*` / `Sensitivity`  
- Run history HTML under `outputs/run_history/` (uploaded Excel names + charts after Fit)

## Tests

```bat
pip install pytest
pytest -q
```

## Models

| ID | Use |
|---|---|
| `M2_midi_gam` | **Default / recommended** |
| `M3_hierarchical_bayes` | Rich bridges only; thin designs auto-fallback |
| `M1_register_dynamic` | When many dynamics×registers exist |
| `M0_global_factor` | Baseline sanity check |
