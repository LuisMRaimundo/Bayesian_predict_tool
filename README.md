# String Technique Transfer (local)

Local research tool for **instrument-agnostic** technique transfer of spectral-density metrics (primarily `EWSD_score_acoustic_balanced`) across violin, viola, cello, and double bass.

**Local folder only — not a GitHub project.**

See [METHODOLOGY.md](METHODOLOGY.md) for the scientific protocol.  
Full math / algorithms / code-line map + tutorial: **[TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md)** (open in [StackEdit](https://stackedit.io); LaTeX twin in `docs/TECHNICAL_MANUAL.tex`).  
Reproduce: [REPRODUCE.md](REPRODUCE.md) · Literature priors: [LITERATURE_ALIGNMENT.md](LITERATURE_ALIGNMENT.md)

**Run history:** every Preflight / Fit & predict writes a timestamped folder under `outputs/run_history/` (`RUN_REPORT.md` + `run_manifest.json` + CSV snapshots). Browse `outputs/run_history/INDEX.md`.

## Quick start (GUI)

```bat
cd /d "C:\Users\lmr20\Desktop\Bayesian Tool"
python run_gui.py
```

1. Add bridge files (ordinario + special techniques)  
2. Choose Zenodo ordinario workbook + source (**`MEDIA`** default = `Violin_Media` columns **M/N/O** = Media pp/mf/**ff**; or `ORCH` / `IOWA`)  
3. Click **Preflight**  
4. Keep **Strict dynamics** ON; model **M2**  
5. **Fit & predict** → use this triad (also highlighted yellow in the Excel `README` sheet):

   1. **File:** your output workbook (e.g. `outputs\transfer_audit.xlsx`)  
   2. **Page:** **`Predictions_supported`**  
   3. **Column:** **`y_pred`**  ← values that mimic the special technique on IOWA/ORCHIDEA

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
