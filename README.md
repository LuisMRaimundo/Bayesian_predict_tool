# String Technique Transfer (local)

Local research tool for **instrument-agnostic** technique transfer of spectral-density metrics (primarily `EWSD_score_acoustic_balanced`) across violin, viola, cello, and double bass.

**Local folder only — not a GitHub project.**

See [METHODOLOGY.md](METHODOLOGY.md) for the scientific protocol.

## Quick start (GUI)

```bat
cd /d "C:\Users\lmr20\Desktop\Bayesian Tool"
python run_gui.py
```

1. Add bridge files (ordinario + special techniques)  
2. Choose Zenodo ordinario workbook + collection (`ORCH` / `IOWA` / `BOTH`)  
3. Click **Preflight**  
4. Keep **Strict dynamics** ON; model **M2**  
5. **Fit & predict** → use this triad (also highlighted yellow in the Excel `README` sheet):

   1. **File:** your output workbook (e.g. `outputs\transfer_audit.xlsx`)  
   2. **Page:** **`Predictions_supported`**  
   3. **Column:** **`y_pred`**  ← values that mimic the special technique on IOWA/ORCHIDEA

## CLI

```bat
python -m string_technique_transfer.cli --bridge data\violin_bridge_panel.csv --target "C:\Users\lmr20\Desktop\VIOLIN_Zenodo_collections_Arco_normal.xlsx" --zenodo-collection ORCH --model M2_midi_gam --out outputs\transfer_audit.xlsx

python -m string_technique_transfer.cli --bridge ... --target ... --preflight-only
```

## Robustness features

- Zenodo triad awareness (`pp/mf/ff`) with tight adequate-dynamic map  
- Winsorize + acoustic shrink/clip  
- No spline extrapolation outside bridge MIDI  
- Preflight gate + blocked pitch-region CV  
- Excel sheets: `Preflight`, `Quality_report`, `Blocked_CV`, `Predictions_supported`, `Predictions_all`

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
