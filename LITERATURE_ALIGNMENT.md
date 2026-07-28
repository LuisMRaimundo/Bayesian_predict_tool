# Literature alignment (acoustics)

This tool’s soft priors and scientific labels are aligned with the STE documentation under:

- `C:\Users\lmr20\Desktop\Extrapolação de ponticelo, sord, suç tasto`
- `...\literature\` and `...\reports\` (mute model, qualitative constraints, Meyer curator notes)

## Binding rules taken from that corpus

1. **No activated universal EWSD technique coefficients** from secondary synthesis / Meyer / Evangelista tables.
2. **Mute physics** is frequency-dependent: \(S_{\mathrm{muted}}(f)=S_{\mathrm{ord}}(f)\,A_{m,i}(f)\).  
   This package approximates that with a **scalar log-ratio** on a density metric (`EWSD_score_acoustic_balanced` / Zenodo Media).
3. **dB / loudness / bridge-mobility** figures are **level proxies**, not density multipliers.
4. **Heavy practice mutes** are out of scope; priors target standard performance `con_sordino`.
5. Qualitative directions (mute/tasto often decrease brilliance-related energy; ponticello often increases upper-partial energy / variance) inform **soft bounds only**.
6. Predictions remain **`model_derived_synthetic` / transport priors**, never relabeled as measured collection data.

## Where this appears in code

| Rule | Location |
|------|----------|
| Soft prior table | `string_technique_transfer/acoustics.py` (`TECHNIQUE_PRIOR`) |
| Module docstring | same file |
| Excel acoustic policy text | `export/excel_audit.py` Formulas sheet |
| Technical manual | `TECHNICAL_MANUAL.md` |
