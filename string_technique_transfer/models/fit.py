"""Fit M0–M3 technique-transfer models and predict onto target ordinario curves."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

from .base import MODEL_CHOICES, FitResult


def _bayes_stack():
    """Lazy import so GUI/CLI stay fast when PyMC is unused."""
    try:
        import bambi as bmb  # type: ignore
        import arviz as az  # type: ignore

        return bmb, az
    except Exception:
        return None, None


def _register_bin(midi: float) -> str:
    if pd.isna(midi):
        return "unknown"
    m = float(midi)
    if m < 55:
        return "low"
    if m < 69:
        return "middle"
    if m < 84:
        return "high"
    return "very_high"


def fit_model(bridge: pd.DataFrame, model_id: str = "M2_midi_gam", metric: str | None = None) -> FitResult:
    if model_id not in MODEL_CHOICES:
        raise ValueError(f"Unknown model_id {model_id}. Choose from {MODEL_CHOICES}")
    df = bridge.copy()
    df = df.dropna(subset=["log_ratio", "midi", "technique"])
    metric = metric or (str(df["metric"].iloc[0]) if "metric" in df.columns else "EWSD_score_acoustic_balanced")
    df["register"] = df["midi"].map(_register_bin)
    df["dynamic"] = df["dynamic"].fillna("unspecified").astype(str)
    df["technique"] = df["technique"].astype(str)
    df["corpus_id"] = df.get("corpus_id", pd.Series(["unknown"] * len(df))).astype(str)

    if model_id == "M0_global_factor":
        return _fit_m0(df, metric)
    if model_id == "M1_register_dynamic":
        return _fit_m1(df, metric)
    if model_id == "M2_midi_gam":
        return _fit_m2(df, metric)
    return _fit_m3(df, metric)


def _fit_m0(df: pd.DataFrame, metric: str) -> FitResult:
    rows = []
    for tech, g in df.groupby("technique"):
        mu = float(g["log_ratio"].mean())
        se = float(g["log_ratio"].sem(ddof=1)) if len(g) > 1 else 0.5
        rows.append({"technique": tech, "dynamic": "all", "register": "all", "log_effect": mu, "se": se, "n": len(g)})
    effects = pd.DataFrame(rows)
    return FitResult(
        model_id="M0_global_factor",
        backend="empirical_mean",
        metric=metric,
        bridge_n=len(df),
        effects=effects,
        diagnostics={"techniques": int(df["technique"].nunique())},
    )


def _fit_m1(df: pd.DataFrame, metric: str) -> FitResult:
    rows = []
    for (tech, dyn, reg), g in df.groupby(["technique", "dynamic", "register"]):
        mu = float(g["log_ratio"].mean())
        se = float(g["log_ratio"].sem(ddof=1)) if len(g) > 1 else 0.5
        rows.append(
            {
                "technique": tech,
                "dynamic": dyn,
                "register": reg,
                "log_effect": mu,
                "se": max(se, 0.05),
                "n": len(g),
            }
        )
    # partial pooling toward technique mean for thin cells
    tech_mean = df.groupby("technique")["log_ratio"].mean().to_dict()
    tech_n = df.groupby("technique")["log_ratio"].count().to_dict()
    eff = pd.DataFrame(rows)
    pooled = []
    for _, r in eff.iterrows():
        n = r["n"]
        prior_n = 4.0
        mu0 = tech_mean[r["technique"]]
        mu = (n * r["log_effect"] + prior_n * mu0) / (n + prior_n)
        se = float(np.sqrt(1.0 / (n + prior_n)) * max(df.loc[df["technique"] == r["technique"], "log_ratio"].std(ddof=1), 0.2))
        pooled.append({**r.to_dict(), "log_effect": float(mu), "se": se, "pooled": True})
    effects = pd.DataFrame(pooled)
    return FitResult(
        model_id="M1_register_dynamic",
        backend="partial_pooling",
        metric=metric,
        bridge_n=len(df),
        effects=effects,
        diagnostics={"cells": len(effects), "tech_means": {k: float(v) for k, v in tech_mean.items()}},
        params={"tech_n": {k: int(v) for k, v in tech_n.items()}},
    )


def _fit_m2(df: pd.DataFrame, metric: str) -> FitResult:
    """Regularized MIDI-smooth transfer model per technique."""
    from ..acoustics import robust_log_weights, shrink_log_ratio

    models: dict[str, Any] = {}
    effect_rows = []
    midi_range: dict[str, tuple[float, float]] = {}
    for tech, g in df.groupby("technique"):
        g = g.copy()
        midi_range[tech] = (float(g["midi"].min()), float(g["midi"].max()))
        mu_med = float(g["log_ratio"].median())
        mu = shrink_log_ratio(mu_med, str(tech), n_eff=float(len(g)))
        se = float(g["log_ratio"].sem(ddof=1)) if len(g) > 1 else 0.35
        se = max(se, 0.12)

        # Small-n or single-dynamic: robust constant (more stable than wiggly spline)
        use_constant = len(g) < 12 or g["dynamic"].nunique() == 1 and len(g) < 25
        if use_constant:
            models[tech] = {
                "type": "constant",
                "mu": mu,
                "se": se,
                "midi_min": midi_range[tech][0],
                "midi_max": midi_range[tech][1],
            }
            effect_rows.append(
                {
                    "technique": tech,
                    "dynamic": "all",
                    "register": "all",
                    "log_effect": mu,
                    "se": se,
                    "n": len(g),
                    "fit_type": "robust_constant",
                }
            )
            continue
        try:
            df_spline = max(3, min(4, len(g) // 6))
            y = g["log_ratio"].to_numpy()
            if g["dynamic"].nunique() > 1:
                X = dmatrix(
                    f"bs(midi, df={df_spline}, include_intercept=True) + C(dynamic)",
                    g,
                    return_type="dataframe",
                )
            else:
                X = dmatrix(
                    f"bs(midi, df={df_spline}, include_intercept=True)",
                    g,
                    return_type="dataframe",
                )
            w = robust_log_weights(y, g["se_log_obs"].to_numpy() if "se_log_obs" in g else None)
            fit = sm.WLS(y, X, weights=w).fit()
            # shrink fitted median toward prior
            med_fit = float(np.median(fit.fittedvalues))
            med_fit = shrink_log_ratio(med_fit, str(tech), n_eff=float(len(g)))
            resid_sd = float(np.std(fit.resid, ddof=1)) if len(g) > 2 else se
            models[tech] = {
                "type": "gam",
                "result": fit,
                "design_info": X.design_info,
                "resid_sd": max(resid_sd, 0.12),
                "mu_center": med_fit,
                "midi_min": midi_range[tech][0],
                "midi_max": midi_range[tech][1],
            }
            effect_rows.append(
                {
                    "technique": tech,
                    "dynamic": "all",
                    "register": "all",
                    "log_effect": med_fit,
                    "se": max(float(g["log_ratio"].sem(ddof=1)), 0.12),
                    "n": len(g),
                    "r2": float(getattr(fit, "rsquared", np.nan)),
                    "fit_type": "regularized_gam",
                }
            )
        except Exception as exc:  # noqa: BLE001
            models[tech] = {
                "type": "constant",
                "mu": mu,
                "se": se,
                "error": str(exc),
                "midi_min": midi_range[tech][0],
                "midi_max": midi_range[tech][1],
            }
            effect_rows.append(
                {
                    "technique": tech,
                    "dynamic": "all",
                    "register": "all",
                    "log_effect": mu,
                    "se": se,
                    "n": len(g),
                    "fit_type": "constant_after_gam_fail",
                }
            )

    # Cross-corpus transport sd; floor higher when all pairs are transport priors
    if df["corpus_id"].nunique() > 1:
        transport_sd = float(df.groupby("corpus_id")["log_ratio"].mean().std(ddof=0))
    else:
        transport_sd = 0.18
    if "is_transport_prior" in df.columns and float(df["is_transport_prior"].mean()) > 0.5:
        transport_sd = max(transport_sd, 0.22)
    transport_sd = max(transport_sd, 0.12)

    return FitResult(
        model_id="M2_midi_gam",
        backend="regularized_robust_transfer",
        metric=metric,
        bridge_n=len(df),
        effects=pd.DataFrame(effect_rows),
        params={"models": models, "transport_sd": transport_sd, "midi_range": midi_range},
        diagnostics={
            "transport_sd": transport_sd,
            "techniques_fit": list(models),
            "policy": "adequate_dynamics+winsor+acoustic_shrink",
        },
    )


def _fit_m3(df: pd.DataFrame, metric: str) -> FitResult:
    # Skip full Bayes when the design is too thin — avoids noisy Windows/PyTensor failures
    n = len(df)
    n_tech = int(df["technique"].nunique())
    n_dyn = int(df["dynamic"].nunique())
    if n < 25 or n_tech < 2:
        approx = _fit_m3_approx(df, metric)
        approx.diagnostics["note"] = (
            f"Full Bayes skipped (n={n}, n_technique={n_tech}, n_dynamic={n_dyn}); "
            "used hierarchical approximation on top of regularized M2."
        )
        approx.backend = "hierarchical_approx_thin_design"
        return approx

    bmb, az = _bayes_stack()
    if bmb is not None and az is not None:
        try:
            return _fit_m3_bambi(df, metric, bmb=bmb, az=az)
        except Exception as exc:  # noqa: BLE001
            approx = _fit_m3_approx(df, metric)
            approx.diagnostics["bayes_fallback_reason"] = str(exc)
            approx.diagnostics["note"] = (
                "Full Bambi/PyMC fit failed; used hierarchical approximation. "
                f"Reason: {exc}"
            )
            approx.backend = "hierarchical_approx_after_bayes_fail"
            return approx
    return _fit_m3_approx(df, metric)


def _fit_m3_approx(df: pd.DataFrame, metric: str) -> FitResult:
    """Corpus-level partial pooling + MIDI smooth (no PyMC required)."""
    base = _fit_m2(df, metric)
    corpus_effects = (
        df.groupby("corpus_id")["log_ratio"].agg(n="count", mean="mean", sd="std").reset_index().fillna(0.3)
    )
    grand = float(df["log_ratio"].mean())
    tau = float(corpus_effects["mean"].std(ddof=0)) if len(corpus_effects) > 1 else 0.15
    tau = max(tau, 0.08)
    corpus_effects["pooled_mean"] = corpus_effects.apply(
        lambda r: (r["n"] * r["mean"] + (1 / tau**2) * grand) / (r["n"] + 1 / tau**2)
        if np.isfinite(r["n"])
        else grand,
        axis=1,
    )
    base.model_id = "M3_hierarchical_bayes"
    base.backend = "hierarchical_approx_no_pymc"
    base.params["corpus_effects"] = corpus_effects
    base.params["tau_corpus"] = tau
    base.diagnostics["note"] = (
        "PyMC/Bambi not installed; used hierarchical approximation. "
        "Install optional bayes extras for full Bayesian GAM."
    )
    return base


def _fit_m3_bambi(df: pd.DataFrame, metric: str, *, bmb, az) -> FitResult:
    import os
    import warnings

    # Silence noisy Windows compiler probes; Python backend is fine for small n
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

    data = df.copy()
    data["se"] = data["se_log_obs"].fillna(0.25).clip(lower=0.05)
    n_tech = int(data["technique"].nunique())
    n_dyn = int(data["dynamic"].nunique())
    n = len(data)
    # Adaptive formula: tiny bridges cannot support full technique:bs(midi)+dynamic
    if n < 20 or n_tech == 1:
        # Bambi rejects 0+technique with a single category
        if n_tech == 1 and n_dyn <= 1:
            formula = "log_ratio ~ 1"
        elif n_tech == 1:
            formula = "log_ratio ~ 1 + C(dynamic)"
        elif n_dyn > 1:
            formula = "log_ratio ~ 0 + technique + C(dynamic)"
        else:
            formula = "log_ratio ~ 0 + technique"
        draws, tune, chains = 300, 300, 2
    elif n < 60:
        formula = "log_ratio ~ 0 + technique + technique:bs(midi, df=3) + C(dynamic)"
        draws, tune, chains = 400, 400, 2
    else:
        formula = "log_ratio ~ 0 + technique + technique:bs(midi, df=4) + technique:C(dynamic)"
        draws, tune, chains = 500, 500, 2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        model = bmb.Model(formula, data, family="t")
        idata = model.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            progressbar=False,
        )
    summary = az.summary(idata, var_names=["~mu"], filter_vars="like")
    transport_sd = (
        float(data.groupby("corpus_id")["log_ratio"].mean().std(ddof=0))
        if data["corpus_id"].nunique() > 1
        else 0.15
    )
    # Point prediction uses the regularized M2 surface (stable on Windows).
    # Bambi InferenceData is retained for posterior diagnostics / audit.
    m2 = _fit_m2(df, metric)
    return FitResult(
        model_id="M3_hierarchical_bayes",
        backend="bambi_pymc",
        metric=metric,
        bridge_n=len(data),
        effects=summary.reset_index().rename(columns={"index": "parameter"}),
        params={
            "idata": idata,
            "model": model,
            "models": m2.params.get("models", {}),
            "transport_sd": max(transport_sd, float(m2.params.get("transport_sd", 0.08))),
            "midi_range": m2.params.get("midi_range", {}),
            "formula": formula,
            "data_columns": list(data.columns),
            "point_predict": "M2_regularized_surface",
        },
        diagnostics={
            "divergences": int(idata.sample_stats["diverging"].sum().item())
            if "diverging" in idata.sample_stats
            else None,
            "transport_sd": max(transport_sd, 0.08),
            "formula": formula,
            "point_predict": "M2_regularized_surface",
            "note": (
                "Full Bayes posterior stored in params['idata']; "
                "y_pred uses attached regularized M2 models for numerical stability."
            ),
        },
    )


def _effect_from_fit(fit: FitResult, technique: str, dynamic: str, midi: float) -> tuple[float, float, str]:
    """Return (log_effect, se, model_flag)."""
    from ..acoustics import clip_log_effect

    reg = _register_bin(midi)
    if fit.model_id == "M0_global_factor":
        row = fit.effects.loc[fit.effects["technique"] == technique]
        if row.empty:
            return 0.0, 1.0, "unsupported_technique"
        delta, _ = clip_log_effect(float(row.iloc[0]["log_effect"]), technique)
        return delta, float(row.iloc[0]["se"]), "global_factor"

    if fit.model_id == "M1_register_dynamic":
        rows = fit.effects[
            (fit.effects["technique"] == technique)
            & (fit.effects["dynamic"] == dynamic)
            & (fit.effects["register"] == reg)
        ]
        flag = "register_dynamic"
        if rows.empty:
            rows = fit.effects[(fit.effects["technique"] == technique) & (fit.effects["dynamic"] == dynamic)]
            flag = "dynamic_only"
        if rows.empty:
            rows = fit.effects[fit.effects["technique"] == technique]
            flag = "technique_pooled"
        if rows.empty:
            return 0.0, 1.0, "unsupported_technique"
        delta, _ = clip_log_effect(float(rows.iloc[0]["log_effect"]), technique)
        return delta, float(rows.iloc[0]["se"]), flag

    models = fit.params.get("models", {})
    transport = float(fit.params.get("transport_sd", 0.18))
    m = models.get(technique)
    if m is None:
        return 0.0, 1.0, "unsupported_technique"

    midi_min = m.get("midi_min")
    midi_max = m.get("midi_max")
    in_range = True
    if midi_min is not None and midi_max is not None:
        in_range = float(midi_min) <= float(midi) <= float(midi_max)

    if m.get("type") == "constant":
        delta, _ = clip_log_effect(float(m["mu"]), technique)
        se = float(np.sqrt(m["se"] ** 2 + transport**2))
        flag = "robust_constant" if in_range else "register_extrapolation"
        if not in_range:
            se = float(np.sqrt(se**2 + 0.15**2))
        return delta, se, flag

    if m.get("type") == "gam":
        new = pd.DataFrame({"midi": [midi], "dynamic": [dynamic]})
        try:
            if in_range:
                Xnew = dmatrix(m["design_info"], new, return_type="dataframe")
                for col in m["result"].model.exog_names:
                    if col not in Xnew.columns:
                        Xnew[col] = 0.0
                Xnew = Xnew[m["result"].model.exog_names]
                pred = float(m["result"].predict(Xnew)[0])
                # blend spline with shrunk center for stability
                center = float(m.get("mu_center", pred))
                pred = 0.7 * pred + 0.3 * center
                flag = "interpolation"
                se = float(np.sqrt(m.get("resid_sd", 0.2) ** 2 + transport**2))
            else:
                # outside bridge MIDI: use shrunk center only (no spline extrapolation)
                pred = float(m.get("mu_center", m["result"].params.mean()))
                flag = "register_extrapolation"
                se = float(np.sqrt(m.get("resid_sd", 0.2) ** 2 + transport**2 + 0.18**2))
            delta, _ = clip_log_effect(pred, technique)
            return delta, se, flag
        except Exception:
            pred = float(m.get("mu_center", 0.0))
            delta, _ = clip_log_effect(pred, technique)
            return delta, float(np.sqrt(0.25**2 + transport**2)), "prediction_fallback"

    if fit.backend == "bambi_pymc":
        return 0.0, float(np.sqrt(0.3**2 + transport**2)), "bambi_requires_predict_api"

    return 0.0, 1.0, "unsupported"


def predict_transfer(
    fit: FitResult,
    target_ordinario: pd.DataFrame,
    techniques: list[str],
    *,
    transport_se_extra: float = 0.0,
    bridge_dynamics_by_technique: dict[str, list[str]] | None = None,
    max_dynamic_distance: int = 1,
    strict_dynamics: bool = True,
) -> pd.DataFrame:
    """Apply Y_hat = Y_ord * exp(delta) with support levels and acoustic clipping."""
    from ..acoustics import clip_log_effect
    from ..dynamics import MAX_ADEQUATE_DISTANCE, map_zenodo_dynamic_to_bridge

    if max_dynamic_distance is None:
        max_dynamic_distance = MAX_ADEQUATE_DISTANCE

    tgt = target_ordinario.copy()
    tgt = tgt.dropna(subset=["value", "midi"])
    if "metric" in tgt.columns and (tgt["metric"] == fit.metric).any():
        tgt = tgt.loc[tgt["metric"] == fit.metric]

    # Target outlier threshold (per collection, if available)
    outlier_cut = {}
    if "collection" in tgt.columns:
        for coll, g in tgt.groupby("collection"):
            med = float(g["value"].median())
            q95 = float(g["value"].quantile(0.95))
            outlier_cut[str(coll)] = max(q95, 3.0 * med)
    else:
        med = float(tgt["value"].median())
        outlier_cut["*"] = max(float(tgt["value"].quantile(0.95)), 3.0 * med)

    rows = []
    for tech in techniques:
        bridge_dyns = []
        if bridge_dynamics_by_technique and tech in bridge_dynamics_by_technique:
            bridge_dyns = bridge_dynamics_by_technique[tech]
        for _, r in tgt.iterrows():
            y0 = float(r["value"])
            if y0 <= 0 or not np.isfinite(y0):
                continue
            target_dyn = str(r.get("dynamic", "unspecified"))
            from ..dynamics import is_adequate_dynamic_pair

            if bridge_dyns:
                effect_dyn, dyn_flag, dyn_dist = map_zenodo_dynamic_to_bridge(target_dyn, bridge_dyns)
            else:
                effect_dyn, dyn_flag, dyn_dist = target_dyn, "dynamic_exact", 0.0

            dyn_ok = is_adequate_dynamic_pair(target_dyn, effect_dyn, max_dynamic_distance)
            if strict_dynamics and not dyn_ok:
                # still record as extrapolated row for diagnostics
                pass

            delta, se, model_flag = _effect_from_fit(fit, tech, effect_dyn, float(r["midi"]))
            delta, clipped = clip_log_effect(delta, tech)

            # Inflate uncertainty for weaker support
            if not dyn_ok:
                se = float(np.sqrt(se**2 + 0.25**2))
            if "register_extrapolation" in model_flag:
                se = float(np.sqrt(se**2 + 0.12**2))
            se = float(np.sqrt(se**2 + transport_se_extra**2))

            lo, hi = r.get("ci_low"), r.get("ci_high")
            if pd.notna(lo) and pd.notna(hi) and lo > 0 and hi > 0:
                se_y = (np.log(float(hi)) - np.log(float(lo))) / (2 * 1.96)
                se = float(np.sqrt(se**2 + se_y**2))

            coll = str(r.get("collection", "*"))
            cut = outlier_cut.get(coll, outlier_cut.get("*", np.inf))
            is_outlier = bool(y0 > cut)

            if dyn_ok and model_flag in {"robust_constant", "interpolation", "global_factor", "register_dynamic", "dynamic_only", "technique_pooled"}:
                support_level = "supported"
            elif dyn_ok and model_flag == "register_extrapolation":
                support_level = "extrapolated_register"
            elif not dyn_ok:
                support_level = "extrapolated_dynamic"
            else:
                support_level = "extrapolated"

            if is_outlier:
                se = float(np.sqrt(se**2 + 0.20**2))
                support_level = (
                    "supported_outlier_target"
                    if support_level == "supported"
                    else support_level + "+outlier_target"
                )

            yhat = y0 * np.exp(delta)
            rows.append(
                {
                    "instrument": r.get("instrument"),
                    "collection": r.get("collection"),
                    "technique": tech,
                    "dynamic": target_dyn,
                    "bridge_dynamic_used": effect_dyn,
                    "dynamic_match": dyn_flag,
                    "dynamic_distance": dyn_dist,
                    "dynamic_adequate": dyn_ok,
                    "midi": r.get("midi"),
                    "note": r.get("note"),
                    "metric": fit.metric,
                    "y_ordinario": y0,
                    "target_outlier": is_outlier,
                    "log_effect": delta,
                    "factor": float(np.exp(delta)),
                    "factor_clipped": clipped,
                    "y_pred": float(yhat),
                    "y_pred_lo95": float(yhat * np.exp(-1.96 * se)),
                    "y_pred_hi95": float(yhat * np.exp(1.96 * se)),
                    "combined_se_log": se,
                    "model_flag": model_flag,
                    "support_level": support_level,
                    "support_flag": f"{model_flag}|{dyn_flag}",
                    "estimate_class": "model_derived_synthetic",
                    "model_id": fit.model_id,
                    "backend": fit.backend,
                }
            )
    return pd.DataFrame(rows)
