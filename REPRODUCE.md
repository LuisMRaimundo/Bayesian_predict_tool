# Reproduce Media-ff sordina-style transfer

## 1. Install

```bat
cd /d "C:\Users\lmr20\Desktop\Bayesian Tool"
pip install -e .
pip install pytest
```

Optional full Bayes:

```bat
pip install -e ".[bayes]"
```

## 2. Smoke (no external data)

```bat
python examples\smoke_transfer.py
pytest -q
```

## 3. Violin Media-ff transfer (your Zenodo workbook)

```bat
python -m string_technique_transfer.cli ^
  --bridge data\violin_bridge_panel.csv ^
  --target "C:\Users\lmr20\Desktop\VIOLIN_Zenodo_collections_Arco_normal.xlsx" ^
  --zenodo-collection MEDIA ^
  --model M2_midi_gam ^
  --out outputs\transfer_audit.xlsx
```

Or GUI: `python run_gui.py` → collection **MEDIA** → Fit & predict.

## 4. What to open

1. File: `outputs\transfer_audit.xlsx`  
2. Page: `Predictions_supported`  
3. Column: `y_pred` (yellow)

Also inspect: `Model_comparison`, `Calibration`, `Holdout_summary`, `Blocked_CV`.

**Run history:** open the path printed as `Run history HTML:` (under `outputs\run_history\<id>\RUN_REPORT.html`). It lists every uploaded Excel and charts the transfer (empty charts are normal for `--preflight-only`). Index: `outputs\run_history\INDEX.md`.

## 5. Scientific constraints

See `LITERATURE_ALIGNMENT.md` — soft priors only; no activated universal EWSD mute law from literature.
