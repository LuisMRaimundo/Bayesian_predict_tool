---
title: String Technique Transfer — Technical Manual
subtitle: Mathematical models, algorithms, code map, and tutorial
version: 1.1
tool: Bayesian_predict_tool / string_technique_transfer
stackedit: Open this file in https://stackedit.io (Markdown + KaTeX)
---

# String Technique Transfer — Technical Manual

**Package:** `string_technique_transfer`  
**Repository:** https://github.com/LuisMRaimundo/Bayesian_predict_tool  
**Local folder:** `C:\Users\lmr20\Desktop\Bayesian Tool`  
**How to view equations:** open this file in [StackEdit](https://stackedit.io) (File → Open from disk). All mathematics uses standard LaTeX delimiters (`$$ … $$`, `$…$`).

### v1.2 upgrades (implemented)

- **M3 posterior prediction:** Bambi `predict(..., kind="mean")` on $\mu$; M2 fallback if predict fails.  
- **Calibration:** blocked-CV residual scale + conformal $q_{95}$ half-widths (`validation/calibration.py`).  
- **Model comparison:** M0–M3 blocked CV + `recommended_model` (`validation/compare.py`).  
- **Holdout pack:** random MIDI holdout + optional external measured scorer (`validation/holdout.py`).  
- **Corpus pooling:** applied to M3-approx technique centers (`_fit_m3_approx`).  
- **Sensitivity:** winsor × acoustic-prior grid (`validation/sensitivity.py`).  
- **Bridge helper:** `bridge_build.py`.  
- **CI:** `.github/workflows/ci.yml` + `examples/smoke_transfer.py`.  
- **Literature:** `LITERATURE_ALIGNMENT.md` (soft priors only; mute $A(f)$ not a universal EWSD law).

---

## Contents

1. [Pedagogical tutorial](#1-pedagogical-tutorial)
2. [Scientific goal and notation](#2-scientific-goal-and-notation)
3. [End-to-end pipeline](#3-end-to-end-pipeline)
4. [Bridge construction](#4-bridge-construction)
5. [Acoustic priors and robustification](#5-acoustic-priors-and-robustification)
6. [Dynamics matching](#6-dynamics-matching)
7. [Models M0–M3](#7-models-m0m3)
8. [Prediction and uncertainty](#8-prediction-and-uncertainty)
9. [Blocked pitch-region CV](#9-blocked-pitch-region-cv)
10. [Code map (file → lines → formulas)](#10-code-map-file--lines--formulas)
11. [Consistency audit](#11-consistency-audit)
12. [What to publish](#12-what-to-publish)

---

## 1. Pedagogical tutorial

### 1.1 The problem in one sentence

You have a **measured ordinario** spectral-density curve (IOWA / ORCHIDEA / Zenodo Media mean) and a **small paired bridge** where both ordinario and a special technique (e.g. con sordino) were measured. You want a **model-derived synthetic** curve for the special technique on the target pitches/dynamics — **not** a claim that those values were measured in the collection.

### 1.2 The core idea (transport, not curve replacement)

Work on the **log-multiplicative** scale:

$$
\delta_{c,t} = \log\frac{Y_{c,t}}{Y_{c,\mathrm{ord}}}
\qquad\Rightarrow\qquad
\widehat Y_{c,t} = Y_{c,\mathrm{ord}}\cdot e^{\widehat\delta_{c,t}}.
$$

Here $c$ indexes a pitch (MIDI / note), $t$ a technique, and $Y$ is the metric (default: `EWSD_score_acoustic_balanced`, Zenodo **Media** column for averaged IOWA/ORCH).

**Intuition:** if sordina is typically $\sim 85\%$ of ordinario on the bridge, then on a new ordinario profile you multiply by $\approx 0.85$, with uncertainty and support flags.

### 1.3 Three ingredients you must feed the tool

| Ingredient | Role | Example |
|---|---|---|
| **Bridge** | Paired ordinario + special technique | `sordina_forte.xlsx` + `Arco_Normal_forte.xlsx` |
| **Target ordinario** | Curve to transport onto | `VIOLIN_Zenodo_collections_Arco_normal.xlsx` with source **MEDIA** |
| **Model** | How $\delta$ varies with pitch/dynamic | Prefer **M2** |

### 1.4 Recommended workflow (GUI)

1. Run `python run_gui.py` from the tool folder.  
2. Add bridge files (ordinario + special techniques).  
3. Choose Zenodo workbook; set collection to **MEDIA** (reads `Violin_Media` columns **M/N/O** = Media pp/mf/**ff**).  
4. Keep **Strict dynamics** ON; model **M2**.  
5. Click **Preflight**, then **Fit & predict**.  
6. Open the Excel audit and use:

| # | Item | Value |
|---|---|---|
| 1 | **File** | your output `.xlsx` (e.g. `transfer_audit.xlsx`) |
| 2 | **Page** | `Predictions_supported` |
| 3 | **Column** | `y_pred` (yellow) |

### 1.5 CLI quick start

```bat
cd /d "C:\Users\lmr20\Desktop\Bayesian Tool"
python -m string_technique_transfer.cli ^
  --bridge data\violin_bridge_panel.csv ^
  --target "C:\Users\lmr20\Desktop\VIOLIN_Zenodo_collections_Arco_normal.xlsx" ^
  --zenodo-collection MEDIA ^
  --model M2_midi_gam ^
  --out outputs\transfer_audit.xlsx
```

### 1.6 How to read support levels

| `support_level` | Meaning | Use in papers? |
|---|---|---|
| `supported` | Adequate dynamic + in-bridge register | **Yes** (as transport prior) |
| `supported_outlier_target` | Supported but ordinario looks extreme | Yes, with flag |
| `extrapolated_dynamic` | e.g. bridge $f$ → Zenodo $pp$ | Exploratory only |
| `extrapolated_register` | MIDI outside bridge range | Exploratory only |

### 1.7 Worked mental example (con sordino @ ff)

1. Bridge pairs give raw ratios $Y_{\mathrm{sord}}/Y_{\mathrm{ord}}$ near MIDI 55–91.  
2. After winsorize + acoustic shrink, M2 fits a **robust constant**  
   $\widehat\delta \approx \log(0.85)$.  
3. Zenodo **Media ff** (column O) supplies $Y_{\mathrm{ord}}(c)$.  
4. Output: $\widehat Y_{\mathrm{sord}}(c)=Y_{\mathrm{ord}}(c)\,e^{\widehat\delta}$.  
5. Only ff rows are **supported** if the bridge is forte-only ($f\leftrightarrow ff$).

---

## 2. Scientific goal and notation

### 2.1 Quantity of interest

Let $Y(i,k,t,d,m)$ be a spectral-density score for instrument $i$, collection/corpus $k$, technique $t$, dynamic $d$, MIDI pitch $m$.

Default metric name in code: `EWSD_score_acoustic_balanced`.  
On Zenodo Media sheet, the ff baseline is column **O** (`Media ff`).

### 2.2 Notation table

| Symbol | Meaning |
|---|---|
| $Y_{\mathrm{ord}}$ | Ordinario metric value |
| $Y_t$ | Special-technique metric value |
| $\delta=\log(Y_t/Y_{\mathrm{ord}})$ | Log-ratio (bridge observation) |
| $\widehat\delta$ | Modelled log-effect |
| $f=e^{\widehat\delta}$ | Multiplicative factor (`factor`) |
| $\widehat Y = Y_{\mathrm{ord}}\,e^{\widehat\delta}$ | Prediction (`y_pred`) |
| $\sigma$ | Combined SE on log scale (`combined_se_log`) |

### 2.3 Scientific label (mandatory)

All predictions are tagged:

$$
\texttt{estimate\_class} = \texttt{model\_derived\_synthetic}.
$$

Never relabel them as measured IOWA/ORCHIDEA observations.

---

## 3. End-to-end pipeline

**File:** `string_technique_transfer/pipeline.py` — function `run_transfer`.

```text
bridge_panel + target_ordinario
        │
        ▼
   preflight_transfer          (preflight.py)
        │
        ▼
   build_log_ratios            (bridge.py)     → δ observations
        │
        ▼
   fit_model                   (models/fit.py) → FitResult (M0–M3)
        │
        ▼
   predict_transfer            (models/fit.py) → Predictions_all
        │
        ├── filter support     → Predictions_supported
        ├── blocked_pitch_cv   (validation/blocked_cv.py)
        └── export_audit_workbook (export/excel_audit.py)
```

---

## 4. Bridge construction

**File:** `string_technique_transfer/bridge.py`  
**Function:** `build_log_ratios` — **lines 19–161**

### 4.1 Pairing rule

For each special-technique row, find ordinario with same instrument and MIDI. Prefer:

1. same collection + same dynamic;  
2. same collection + **adequate** nearest dynamic;  
3. else other collections, still requiring adequate dynamic.

If no adequate dynamic exists, the pair is **skipped**.

### 4.2 Raw log-ratio

$$
\delta^{\mathrm{raw}} = \log\frac{Y_t}{Y_{\mathrm{ord}}},\qquad
Y_{\mathrm{ord}} = \mathrm{median}\{\text{matched ordinario values}\}.
$$

**Code:** `bridge.py` ≈ lines 92–114.

### 4.3 Observation SE from confidence intervals (if present)

$$
\mathrm{se}_{\log}(Y) = \frac{\log Y_{\mathrm{hi}}-\log Y_{\mathrm{lo}}}{2\cdot 1.96},
\qquad
\mathrm{se}_{\mathrm{obs}} = \sqrt{\mathrm{se}_t^2 + \mathrm{se}_{\mathrm{ord}}^2}.
$$

**Code:** `_log_se_from_ci` lines 13–16; combination lines 97–112.

### 4.4 Winsorization (per technique)

With quantile $q=0.05$ (default):

$$
\delta^{\mathrm{w}} = \mathrm{clip}\!\left(\delta^{\mathrm{raw}},\, Q_q(\delta^{\mathrm{raw}}),\, Q_{1-q}(\delta^{\mathrm{raw}})\right).
$$

**Code:** lines 148–154.

### 4.5 Per-row acoustic shrink

Each winsorized ratio is shrunk with $n_{\mathrm{eff}}=1$ (see §5):

$$
\delta = \frac{1\cdot \delta^{\mathrm{w}} + \kappa_t\,\delta_t^{\mathrm{prior}}}{1+\kappa_t}.
$$

**Code:** lines 155–156.

---

## 5. Acoustic priors and robustification

**File:** `string_technique_transfer/acoustics.py`

### 5.1 Technique priors (research soft priors)

| Technique | Prior factor $e^{\delta^{\mathrm{prior}}}$ | Bounds $[f_{\lo},f_{\mathrm{hi}}]$ | $\kappa$ | Lines |
|---|---|---|---|---|
| `con_sordino` | $0.85$ | $[0.45,1.05]$ | $4$ | 10–16 |
| `sul_ponticello` | $1.25$ | $[0.90,2.40]$ | $3$ | 17–23 |
| `sul_tasto` | $0.80$ | $[0.50,1.10]$ | $3$ | 24–30 |
| `natural_harmonics` | $0.65$ | $[0.25,1.05]$ | $4$ | 31–37 |
| `artificial_harmonics` | $0.60$ | $[0.25,1.05]$ | $4$ | 38–44 |

### 5.2 Shrinkage toward prior

**Function:** `shrink_log_ratio` — **lines 48–54**

$$
\widetilde\delta = \frac{n_{\mathrm{eff}}\,\delta + \kappa_t\,\delta_t^{\mathrm{prior}}}{n_{\mathrm{eff}}+\kappa_t}.
$$

### 5.3 Hard clip of effects

**Function:** `clip_log_effect` — **lines 57–66**

$$
\widehat\delta \leftarrow \mathrm{clip}\!\left(\widehat\delta,\,\log f_{\lo},\,\log f_{\mathrm{hi}}\right).
$$

### 5.4 Huber-like weights (M2 GAM)

**Function:** `robust_log_weights` — **lines 69–81**

$$
\mathrm{MAD}=\mathrm{median}_i|\delta_i-\mathrm{median}(\delta)|,
\qquad
z_i=\frac{|\delta_i-\mathrm{median}(\delta)|}{1.4826\,\mathrm{MAD}},
$$

$$
w_i =
\begin{cases}
1, & z_i\le 2.5,\\
2.5/z_i, & z_i>2.5.
\end{cases}
$$

If observation SEs exist, multiply by normalized inverse-variance weights $1/\sigma_i^2$.

---

## 6. Dynamics matching

**File:** `string_technique_transfer/dynamics.py`

### 6.1 Loudness scale

$$
\mathrm{pp}{:}0,\; \mathrm{p}{:}1,\; \mathrm{mp}{:}2,\; \mathrm{mf}{:}3,\; \mathrm{f}{:}4,\; \mathrm{ff}{:}5.
$$

**Code:** `DYNAMIC_LEVEL` lines 15–23.

### 6.2 Adequate Zenodo ↔ bridge map (strict policy)

$$
\begin{align*}
\mathrm{pp} &\leftrightarrow \{\mathrm{pp},\mathrm{p}\},\\
\mathrm{mf} &\leftrightarrow \{\mathrm{mf},\mathrm{mp}\},\\
\mathrm{ff} &\leftrightarrow \{\mathrm{ff},\mathrm{f}\}.
\end{align*}
$$

**Code:** `ADEQUATE_BRIDGE_FOR_ZENODO` lines 33–37;  
`is_adequate_dynamic_pair` **lines 60–71**;  
`map_zenodo_dynamic_to_bridge` **lines 100–114**.

### 6.3 Consequence

A forte-only bridge supports Zenodo **ff** only. Rows for pp/mf are still written to `Predictions_all` as `extrapolated_dynamic`, but must not be used as primary results.

---

## 7. Models M0–M3

**Dispatcher:** `fit_model` — `models/fit.py` **lines 39–56**  
**Choices:** `models/base.py` `MODEL_CHOICES`.

Register bins used by M1 (`_register_bin`, **lines 26–36**):

$$
\begin{align*}
m &< 55 &&\Rightarrow \texttt{low},\\
55\le m &< 69 &&\Rightarrow \texttt{middle},\\
69\le m &< 84 &&\Rightarrow \texttt{high},\\
m &\ge 84 &&\Rightarrow \texttt{very\_high}.
\end{align*}
$$

---

### 7.1 M0 — Global factor

**Function:** `_fit_m0` — **lines 59–73**  
**Backend:** `empirical_mean`  
**When:** sanity baseline.

For each technique $t$:

$$
\widehat\delta_t = \overline{\delta}_{i:\,t_i=t},
\qquad
\mathrm{se}_t = \mathrm{SEM}(\delta_{i:t})
\quad(\text{fallback }0.5\text{ if }n=1).
$$

**Prediction:** same $\widehat\delta_t$ for all MIDI/dynamics (`_effect_from_fit` lines 381–386).

---

### 7.2 M1 — Register × dynamic cells with partial pooling

**Function:** `_fit_m1` — **lines 76–112**  
**Backend:** `partial_pooling`

Cell mean for $(t,d,r)$:

$$
\widehat\delta_{t,d,r}^{\mathrm{cell}} = \overline{\delta}_{t,d,r}.
$$

Pool toward technique mean $\mu_t$ with prior strength $n_0=4$:

$$
\widehat\delta_{t,d,r}
=
\frac{n_{t,d,r}\,\widehat\delta_{t,d,r}^{\mathrm{cell}} + n_0\,\mu_t}{n_{t,d,r}+n_0}.
$$

SE (heuristic):

$$
\mathrm{se}_{t,d,r}
=
\sqrt{\frac{1}{n_{t,d,r}+n_0}}\;
\max\!\big(\mathrm{sd}_t(\delta),\,0.2\big).
$$

**Prediction fallback ladder** (`_effect_from_fit` lines 388–404): exact cell → same dynamic any register → technique-only.

---

### 7.3 M2 — Regularized MIDI transfer (**recommended**)

**Function:** `_fit_m2` — **lines 115–236**  
**Backend:** `regularized_robust_transfer`  
**Point prediction:** `_effect_from_fit` lines 406–451.

#### Step A — robust center

$$
\mu_t^{\mathrm{med}}=\mathrm{median}(\delta_{i:t}),
\qquad
\mu_t=\mathrm{shrink}(\mu_t^{\mathrm{med}},t,n_{\mathrm{eff}}=n_t).
$$

#### Step B — choose constant vs GAM

Use **robust constant** if

$$
n_t < 12
\quad\text{or}\quad
\big(\#\{\text{dynamics}\}=1 \;\wedge\; n_t<25\big).
$$

Then store $\mu_t$ with $\mathrm{se}_t=\max(\mathrm{SEM},0.12)$.

#### Step C — regularized GAM (richer bridges)

Design matrix (Patsy B-spline):

$$
\delta_i \approx \mathbf{x}(m_i,d_i)^\top\boldsymbol\beta,
\qquad
\mathbf{x}=
\begin{cases}
\mathrm{bs}(m,\mathrm{df})+C(d), & \# d>1,\\
\mathrm{bs}(m,\mathrm{df}), & \text{else},
\end{cases}
$$

with $\mathrm{df}=\mathrm{clip}(\lfloor n/6\rfloor,3,4)$.

Fit **WLS** with Huber weights $w_i$ (§5.4):

$$
\widehat{\boldsymbol\beta}
=
\arg\min_{\boldsymbol\beta}
\sum_i w_i\big(\delta_i-\mathbf{x}_i^\top\boldsymbol\beta\big)^2.
$$

Center the fit:

$$
\mu_t^{\mathrm{center}}
=
\mathrm{shrink}\!\big(\mathrm{median}(\widehat\delta_i^{\mathrm{fit}}),\,t,\,n_t\big).
$$

#### Step D — transport SD

$$
\sigma_{\mathrm{tr}}
=
\max\!\Big(
  \mathrm{sd}_k\big(\overline{\delta}_k\big),\;
  0.12
\Big),
$$

floored to $\ge 0.22$ if $>50\%$ of pairs are transport priors.

#### Prediction (constant mode)

$$
\widehat\delta(m)=\mathrm{clip}(\mu_t),\qquad
\sigma=\sqrt{\mathrm{se}_t^2+\sigma_{\mathrm{tr}}^2}
\quad(+\,0.15^2\text{ if }m\notin[m_{\min},m_{\max}]).
$$

#### Prediction (GAM mode, in-range)

$$
\widehat\delta_{\mathrm{raw}}(m,d)=\mathbf{x}(m,d)^\top\widehat{\boldsymbol\beta},
\qquad
\widehat\delta=0.7\,\widehat\delta_{\mathrm{raw}}+0.3\,\mu_t^{\mathrm{center}},
$$

then clip. **Outside** MIDI range: use $\mu_t^{\mathrm{center}}$ only (no spline extrapolation).

---

### 7.4 M3 — Hierarchical Bayes (optional)

**Entry:** `_fit_m3` — **lines 239–266**

#### Gate (thin design)

If $n<25$ or $\#\{\text{techniques}\}<2$: skip full Bayes → `_fit_m3_approx`.

#### 7.4.1 Approximate hierarchy (no PyMC)

**Function:** `_fit_m3_approx` — **lines 269–292**

1. Fit M2 surface (used for $\widehat Y$).  
2. Compute corpus-level means and a diagnostic pooled mean:

$$
\tau=\max\!\big(\mathrm{sd}_k(\bar\delta_k),0.08\big),
\qquad
\bar\delta_k^{\mathrm{pool}}
=
\frac{n_k\,\bar\delta_k + \tau^{-2}\,\bar\delta}{n_k+\tau^{-2}}.
$$

> **Audit note:** $\bar\delta_k^{\mathrm{pool}}$ is stored in `params['corpus_effects']` for audit; **point prediction currently uses the attached M2 surface**, not a second corpus correction. This is intentional for numerical stability.

#### 7.4.2 Full Bambi / PyMC

**Function:** `_fit_m3_bambi` — **lines 295–373**

Student-$t$ family; adaptive formula:

| Design size | Formula (conceptual) |
|---|---|
| tiny | $\delta\sim 1$ or $1+C(d)$ or $0+\mathrm{tech}+C(d)$ |
| medium | $0+\mathrm{tech}+\mathrm{tech}{:}\mathrm{bs}(m,3)+C(d)$ |
| large | $0+\mathrm{tech}+\mathrm{tech}{:}\mathrm{bs}(m,4)+\mathrm{tech}{:}C(d)$ |

**Consistency fix (v1.1):** after sampling, the fit **attaches M2 `models`** for `y_pred`. Posterior `idata` remains for diagnostics. Point predict path is therefore mathematically the M2 surface under an M3 audit wrapper.

---

## 8. Prediction and uncertainty

**Function:** `predict_transfer` — **lines 459–581**

### 8.1 Point prediction

$$
\widehat Y = Y_{\mathrm{ord}}\cdot \exp(\widehat\delta),\qquad
\texttt{factor}=\exp(\widehat\delta).
$$

**Code:** lines 550–569 (`yhat = y0 * np.exp(delta)`).

### 8.2 Combined SE on log scale

Start from model SE, then inflate:

$$
\begin{align*}
\sigma &\leftarrow \sqrt{\sigma^2 + 0.25^2} && \text{inadequate dynamic},\\
\sigma &\leftarrow \sqrt{\sigma^2 + 0.12^2} && \text{register extrapolation},\\
\sigma &\leftarrow \sqrt{\sigma^2 + \sigma_Y^2} && \text{if }Y_{\mathrm{ord}}\text{ has CI},\\
\sigma &\leftarrow \sqrt{\sigma^2 + 0.20^2} && \text{target outlier}.
\end{align*}
$$

### 8.3 Approximate 95% interval

$$
\big[\widehat Y\,e^{-1.96\sigma},\;\widehat Y\,e^{+1.96\sigma}\big]
=
(\texttt{y\_pred\_lo95},\,\texttt{y\_pred\_hi95}).
$$

### 8.4 Target outlier rule

Per collection:

$$
C=\max\!\big(Q_{0.95}(Y),\,3\cdot\mathrm{median}(Y)\big),
\qquad
\mathrm{outlier}\iff Y_{\mathrm{ord}}>C.
$$

**Code:** lines 481–491, 529–531.

---

## 9. Blocked pitch-region CV

**File:** `string_technique_transfer/validation/blocked_cv.py`  
**Function:** `blocked_pitch_cv` — **lines 30–120**

### 9.1 Blocks

Sort unique MIDI; cut a new block when pitch advances by $\ge B$ semitones (default $B=12$).

### 9.2 Fold

For each block $\mathcal{B}$: train on MIDI $\notin\mathcal{B}$, predict $\widehat\delta$ on $\mathcal{B}$ via `_effect_from_fit`.

### 9.3 Metrics

$$
\begin{align*}
\mathrm{MAE}_{\log}&=\mathrm{mean}|\delta-\widehat\delta|,\\
\mathrm{RMSE}_{\log}&=\sqrt{\mathrm{mean}(\delta-\widehat\delta)^2},\\
\mathrm{MAPE}_{f}&=\mathrm{mean}\frac{|e^{\delta}-e^{\widehat\delta}|}{e^{\delta}},\\
\mathrm{bias}_{\log}&=\mathrm{mean}(\delta-\widehat\delta).
\end{align*}
$$

---

## 10. Code map (file → lines → formulas)

| Topic | File | Lines | Formula / algorithm |
|---|---|---|---|
| Pipeline orchestration | `pipeline.py` | `run_transfer` | §3 flowchart |
| Log-ratio bridge | `bridge.py` | 19–161 | $\delta=\log(Y_t/Y_{\mathrm{ord}})$; winsor; shrink |
| CI → log SE | `bridge.py` | 13–16 | $(\log hi-\log lo)/(2\cdot 1.96)$ |
| Factor summary | `bridge.py` | 164–176 | median / IQR of $e^{\delta}$ |
| Priors table | `acoustics.py` | 9–45 | $f^{\mathrm{prior}},[f_{\lo},f_{\mathrm{hi}}],\kappa$ |
| Shrink | `acoustics.py` | 48–54 | $(n\delta+\kappa\delta_0)/(n+\kappa)$ |
| Clip | `acoustics.py` | 57–66 | clip to $\log$ bounds |
| Huber weights | `acoustics.py` | 69–81 | §5.4 |
| Dynamic adequacy | `dynamics.py` | 60–71 | Zenodo map §6.2 |
| Zenodo→bridge map | `dynamics.py` | 100–114 | preferred partner order |
| Register bins | `models/fit.py` | 26–36 | §7 |
| Dispatcher | `models/fit.py` | 39–56 | M0–M3 switch |
| M0 | `models/fit.py` | 59–73 | technique mean |
| M1 | `models/fit.py` | 76–112 | cell + pool $n_0=4$ |
| M2 | `models/fit.py` | 115–236 | constant / WLS B-spline |
| M3 gate | `models/fit.py` | 239–266 | thin-design skip |
| M3 approx | `models/fit.py` | 269–292 | M2 + corpus audit pool |
| M3 Bambi | `models/fit.py` | 295–373 | Student-$t$ GAM + attach M2 |
| Effect lookup | `models/fit.py` | 376–456 | $\widehat\delta(m,d)$ |
| Predict | `models/fit.py` | 459–581 | $Y e^{\delta}$, intervals |
| Blocked CV | `validation/blocked_cv.py` | 30–120 | §9 |
| MEDIA loader (col O) | `io/loaders.py` | `load_zenodo_media_ordinario` | Media pp/mf/ff |
| Excel highlight triad | `export/excel_audit.py` | `PRIMARY_SHEET/COLUMN` | file / page / `y_pred` |
| Model IDs | `models/base.py` | 8–13 | `MODEL_CHOICES` |

---

## 11. Consistency audit

Performed against the implementation in this repository (manual v1.1).

### 11.1 Consistent (verified)

| Claim | Status |
|---|---|
| Transport identity $\widehat Y=Y_{\mathrm{ord}}e^{\widehat\delta}$ | **Consistent** (`predict_transfer`) |
| Bridge $\delta=\log(Y_t/Y_{\mathrm{ord}})$ | **Consistent** |
| Winsorize then shrink then model | **Consistent** order |
| M2 no spline extrapolation outside MIDI | **Consistent** (`register_extrapolation` uses center) |
| Strict dynamics $f\leftrightarrow ff$ only for forte bridges | **Consistent** |
| 95% interval uses $\pm 1.96\sigma$ on log scale | **Consistent** |
| Huber constant $1.4826$ (normal MAD scale) | **Consistent** |
| Blocked CV scores log-effects, reports factor MAPE | **Consistent** |
| MEDIA target uses `Media ff` (Excel column O) | **Consistent** (`load_zenodo_media_ordinario`) |

### 11.2 Design choices / limitations (documented, not bugs)

1. **M3 corpus pooled means** are audit statistics; prediction uses M2 surface.  
2. **Full Bambi M3** stores posterior for diagnostics; **point `y_pred` uses attached M2 models** (stability on Windows / thin designs).  
3. **M1 SE formula** is a heuristic precision weight, not a full Bayesian posterior SD.  
4. **Per-row shrink** in the bridge uses $n_{\mathrm{eff}}=1$; M2 center shrink uses $n_{\mathrm{eff}}=n_t$.  
5. Predictions remain **model-derived synthetics**, even when `supported`.

### 11.3 Fix included in v1.1

Previously, a successful Bambi fit did **not** attach a `models` dict, so `_effect_from_fit` could return `unsupported_technique` / zero effect.  
**Fix:** `_fit_m3_bambi` now attaches the M2 prediction surface (`params['models']`) while keeping `idata`.

---

## 12. What to publish

1. Sheet **`Predictions_supported`**, column **`y_pred`**.  
2. Report: model id (usually M2), bridge $n$, techniques/dynamics, MIDI range, transport-prior fraction, blocked-CV MAE/RMSE.  
3. State clearly: *model-derived synthetic / transport prior*, not measured collection data.  
4. Prefer Zenodo **MEDIA** (`Violin_Media!O` for ff) when comparing to averaged IOWA–ORCH baselines.

---

## Appendix A — Symbol ↔ Excel columns

| Math | Excel column |
|---|---|
| $Y_{\mathrm{ord}}$ | `y_ordinario` |
| $\widehat\delta$ | `log_effect` |
| $e^{\widehat\delta}$ | `factor` |
| $\widehat Y$ | `y_pred` |
| $\widehat Y e^{\pm 1.96\sigma}$ | `y_pred_lo95`, `y_pred_hi95` |
| $\sigma$ | `combined_se_log` |

## Appendix B — Tests

```bat
cd /d "C:\Users\lmr20\Desktop\Bayesian Tool"
pytest -q
```

Covers dynamics, acoustics/bridge shrink, pipeline support split, and Zenodo MEDIA column O.

---

*End of technical manual.*
