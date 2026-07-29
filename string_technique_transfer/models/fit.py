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


def fit_model(
    bridge: pd.DataFrame,
    model_id: str = "M2_midi_gam",
    metric: str | None = None,
    *,
    apply_acoustic_prior: bool = True,
    allow_m3_approx_fallback: bool = False,
    require_paired_corpus_for_m3: bool = False,
    m3_force_approx: bool = False,
) -> FitResult:
    if model_id not in MODEL_CHOICES:
        raise ValueError(f"Unknown model_id {model_id}. Choose from {MODEL_CHOICES}")
    df = bridge.copy()
    df = df.dropna(subset=["log_ratio", "midi", "technique"])
    metric = metric or (str(df["metric"].iloc[0]) if "metric" in df.columns else "EWSD_score_acoustic_balanced")
    df["register"] = df["midi"].map(_register_bin)
    df["dynamic"] = df["dynamic"].fillna("unspecified").astype(str)
    df["technique"] = df["technique"].astype(str)
    if "special_corpus_id" in df.columns:
        df["corpus_id"] = df["special_corpus_id"].astype(str)
    else:
        df["corpus_id"] = df.get("corpus_id", pd.Series(["unknown"] * len(df))).astype(str)

    if model_id == "M0_global_factor":
        return _fit_m0(df, metric)
    if model_id == "M1_register_dynamic":
        return _fit_m1(df, metric)
    if model_id == "M2_midi_gam":
        return _fit_m2(df, metric, apply_acoustic_prior=apply_acoustic_prior)
    return _fit_m3(
        df,
        metric,
        apply_acoustic_prior=apply_acoustic_prior,
        allow_m3_approx_fallback=allow_m3_approx_fallback,
        require_paired_corpus=require_paired_corpus_for_m3,
        force_approx=m3_force_approx,
    )


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


def _fit_m2(
    df: pd.DataFrame, metric: str, *, apply_acoustic_prior: bool = True
) -> FitResult:
    """Regularized MIDI-smooth transfer model per technique."""
    from ..acoustics import robust_log_weights, shrink_log_ratio

    models: dict[str, Any] = {}
    effect_rows = []
    midi_range: dict[str, tuple[float, float]] = {}
    for tech, g in df.groupby("technique"):
        g = g.copy()
        # Exclude invented dynamics from spline factors; keep rows for pitch model
        measured = g[~g["dynamic"].astype(str).str.lower().isin({"unspecified", "unknown"})]
        g_fit = measured if len(measured) >= max(6, len(g) // 3) else g
        midi_range[tech] = (float(g["midi"].min()), float(g["midi"].max()))
        mu_med = float(g_fit["log_ratio"].median())
        n_eff = float(len(g_fit))
        mu = shrink_log_ratio(mu_med, str(tech), n_eff=n_eff) if apply_acoustic_prior else mu_med
        se = float(g_fit["log_ratio"].sem(ddof=1)) if len(g_fit) > 1 else 0.35
        se = max(se, 0.12)
        if "is_transport_prior" in g.columns and float(g["is_transport_prior"].mean()) > 0.5:
            se = max(se, 0.18)

        # Small-n or single measured dynamic: robust constant (more stable than wiggly spline)
        n_dyn_meas = int(g_fit["dynamic"].nunique())
        use_constant = len(g_fit) < 12 or (n_dyn_meas <= 1 and len(g_fit) < 25)
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
                    "n": len(g_fit),
                    "fit_type": "robust_constant",
                }
            )
            continue
        try:
            df_spline = max(3, min(4, len(g_fit) // 6))
            y = g_fit["log_ratio"].to_numpy()
            if n_dyn_meas > 1:
                X = dmatrix(
                    f"bs(midi, df={df_spline}, include_intercept=True) + C(dynamic)",
                    g_fit,
                    return_type="dataframe",
                )
            else:
                X = dmatrix(
                    f"bs(midi, df={df_spline}, include_intercept=True)",
                    g_fit,
                    return_type="dataframe",
                )
            w = robust_log_weights(
                y, g_fit["se_log_obs"].to_numpy() if "se_log_obs" in g_fit else None
            )
            fit = sm.WLS(y, X, weights=w).fit()
            # One coefficient-level shrink of the fitted center (not per-row)
            med_fit = float(np.median(fit.fittedvalues))
            if apply_acoustic_prior:
                med_fit = shrink_log_ratio(med_fit, str(tech), n_eff=n_eff)
            resid_sd = float(np.std(fit.resid, ddof=1)) if len(g_fit) > 2 else se
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
                    "se": max(float(g_fit["log_ratio"].sem(ddof=1)), 0.12),
                    "n": len(g_fit),
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
                    "n": len(g_fit),
                    "fit_type": "constant_after_gam_fail",
                }
            )

    # External transport-uncertainty proxy (NOT identified corpus variance).
    # When corpus ≈ technique (typical Philharmonia folders), SD(corpus means)
    # mixes technique differences with transport error — use only as a floor.
    if "ordinario_corpus_id" in df.columns and "special_corpus_id" in df.columns:
        mismatch = (
            df["ordinario_corpus_id"].astype(str) != df["special_corpus_id"].astype(str)
        ).mean()
    else:
        mismatch = float(df["is_transport_prior"].mean()) if "is_transport_prior" in df.columns else 0.0
    tech_confounded = (
        "technique" in df.columns
        and df.groupby("technique")["corpus_id"].nunique().max() <= 1
        and df["corpus_id"].nunique() > 1
    )
    if tech_confounded or mismatch > 0.5:
        transport_sd = 0.22  # external proxy floor for cross-corpus transport
        transport_sd_source = "external_proxy_technique_corpus_confounded"
    elif df["corpus_id"].nunique() > 1:
        transport_sd = float(df.groupby("corpus_id")["log_ratio"].mean().std(ddof=0))
        transport_sd = max(transport_sd, 0.12)
        transport_sd_source = "dispersion_of_corpus_means_proxy"
    else:
        transport_sd = 0.18
        transport_sd_source = "single_corpus_default_proxy"
    transport_sd = max(transport_sd, 0.12)

    return FitResult(
        model_id="M2_midi_gam",
        backend="regularized_robust_transfer",
        metric=metric,
        bridge_n=len(df),
        effects=pd.DataFrame(effect_rows),
        params={
            "models": models,
            "transport_sd": transport_sd,
            "transport_sd_source": transport_sd_source,
            "midi_range": midi_range,
            "apply_acoustic_prior": apply_acoustic_prior,
            "interval_type": "heuristic_predictive",
        },
        diagnostics={
            "transport_sd": transport_sd,
            "transport_sd_source": transport_sd_source,
            "techniques_fit": list(models),
            "policy": "winsor_responses+coefficient_acoustic_prior"
            if apply_acoustic_prior
            else "winsor_responses_only",
            "interval_type": "heuristic_predictive_not_bayesian_credible",
            "note": (
                "transport_sd is an external transport-uncertainty proxy, "
                "not an empirically identified corpus variance when technique≡corpus."
            ),
        },
    )


def _fit_m3(
    df: pd.DataFrame,
    metric: str,
    *,
    apply_acoustic_prior: bool = True,
    allow_m3_approx_fallback: bool = False,
    require_paired_corpus: bool = False,
    force_approx: bool = False,
) -> FitResult:
    """Fit M3 via heteroscedastic PyMC Student-t (preferred) or authorized approx."""
    from ..paired_corpus import assess_paired_corpus

    n = len(df)
    n_tech = int(df["technique"].nunique())
    n_dyn = int(df["dynamic"].nunique())
    paired = assess_paired_corpus(df)

    def _approx_or_raise(reason: str, backend: str) -> FitResult:
        if not allow_m3_approx_fallback:
            raise RuntimeError(
                "M3 hierarchical Bayes was requested but could not run a real Bayesian fit.\n"
                f"Reason: {reason}\n"
                "Install compatible Bayes extras (`pip install -r requirements-bayes.txt`), or pass "
                "allow_m3_approx_fallback=True to authorize the non-Bayesian approximation "
                "(not equivalent to hierarchical Bayes; not publication-grade)."
            )
        approx = _fit_m3_approx(df, metric, apply_acoustic_prior=apply_acoustic_prior)
        approx.diagnostics["bayes_fallback_reason"] = reason
        approx.diagnostics["note"] = (
            "AUTHORIZED non-Bayesian hierarchical approximation — not posterior sampling. "
            f"Reason: {reason}"
        )
        approx.diagnostics["paired_corpus_tier"] = paired.scientific_tier
        approx.backend = backend
        return approx

    if require_paired_corpus and paired.scientific_tier == "transport_only":
        raise RuntimeError(
            "M3 scientific mode requires at least some same-collection (paired) bridge rows.\n"
            f"{paired.message}\n"
            "Collect ordinario + special technique under the same recording chain, or "
            "disable require_paired_corpus_for_m3 for exploratory transport-only runs."
        )

    # CV / model-comparison: use authorized approx only (avoid multi-fold MCMC)
    if force_approx:
        return _approx_or_raise(
            "force_approx=True (blocked CV / model comparison path)",
            "hierarchical_approx_cv_path",
        )

    if n < 12 or n_tech < 1:
        return _approx_or_raise(
            f"design too thin for full Bayes (n={n}, n_technique={n_tech}, n_dynamic={n_dyn})",
            "hierarchical_approx_thin_design",
        )

    # Preferred path: custom PyMC heteroscedastic Student-t (uses se_log_obs)
    try:
        from .m3_pymc import fit_m3_heteroscedastic

        result = fit_m3_heteroscedastic(
            df,
            metric,
            apply_acoustic_prior=apply_acoustic_prior,
            m2_fallback_fitter=_fit_m2,
            draws=300 if n < 40 else 400,
            tune=300 if n < 40 else 400,
            chains=2,
        )
        result.diagnostics["paired_corpus_tier"] = paired.scientific_tier
        result.diagnostics["paired_fraction"] = paired.paired_fraction
        if paired.scientific_tier == "transport_only":
            result.diagnostics["scientific_caveat"] = paired.message
        return result
    except ImportError as exc:
        pymc_err = str(exc)
    except Exception as exc:  # noqa: BLE001
        pymc_err = str(exc)

    # Legacy Bambi path (homoscedastic; SE not in likelihood) — only if PyMC failed
    bmb, az = _bayes_stack()
    if bmb is not None and az is not None:
        try:
            result = _fit_m3_bambi(
                df, metric, bmb=bmb, az=az, apply_acoustic_prior=apply_acoustic_prior
            )
            result.diagnostics["pymc_heteroscedastic_error"] = pymc_err
            result.diagnostics["note"] = (
                (result.diagnostics.get("note") or "")
                + " | Fell back to Bambi (homoscedastic; SE not in likelihood) after PyMC error: "
                + pymc_err
            )
            result.diagnostics["paired_corpus_tier"] = paired.scientific_tier
            return result
        except Exception as exc:  # noqa: BLE001
            return _approx_or_raise(
                f"PyMC failed ({pymc_err}); Bambi failed ({exc})",
                "hierarchical_approx_after_bayes_fail",
            )

    return _approx_or_raise(
        f"PyMC heteroscedastic fit unavailable: {pymc_err}",
        "hierarchical_approx_no_pymc",
    )


def _fit_m3_approx(
    df: pd.DataFrame, metric: str, *, apply_acoustic_prior: bool = True
) -> FitResult:
    """Corpus-level partial pooling + MIDI smooth (no PyMC required)."""
    base = _fit_m2(df, metric, apply_acoustic_prior=apply_acoustic_prior)
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
    # Wire pooling into the prediction surface: shrink each technique center toward grand
    models = dict(base.params.get("models", {}))
    tech_n = df.groupby("technique")["log_ratio"].count().to_dict()
    for tech, m in models.items():
        n_t = float(tech_n.get(tech, len(df)))
        prec0 = 1.0 / (tau**2)
        if m.get("type") == "constant":
            mu = float(m["mu"])
            m = dict(m)
            m["mu"] = float((n_t * mu + prec0 * grand) / (n_t + prec0))
            m["se"] = float(np.sqrt(m["se"] ** 2 + tau**2))
            m["corpus_pooled"] = True
            models[tech] = m
        elif m.get("type") == "gam":
            m = dict(m)
            center = float(m.get("mu_center", grand))
            m["mu_center"] = float((n_t * center + prec0 * grand) / (n_t + prec0))
            m["resid_sd"] = float(np.sqrt(float(m.get("resid_sd", 0.2)) ** 2 + tau**2))
            m["corpus_pooled"] = True
            models[tech] = m
    base.model_id = "M3_hierarchical_bayes"
    base.backend = "hierarchical_approx_no_pymc"
    base.params["models"] = models
    base.params["corpus_effects"] = corpus_effects
    base.params["tau_corpus"] = tau
    base.params["grand_mean"] = grand
    base.params["corpus_pooling_applied"] = True
    base.params["transport_sd"] = max(float(base.params.get("transport_sd", 0.12)), tau)
    base.diagnostics["note"] = (
        "Hierarchical approximation (NOT Bayesian sampling): M2 surface + technique centers "
        "pooled toward grand mean; corpus_effects stored for audit."
    )
    base.diagnostics["corpus_pooling_applied"] = True
    base.diagnostics["is_bayesian"] = False
    return base


def _corpus_hierarchy_identifiable(data: pd.DataFrame) -> bool:
    """True only if at least one technique appears in ≥2 corpora (not confounded)."""
    if "corpus_id" not in data.columns:
        return False
    return bool((data.groupby("technique")["corpus_id"].nunique() > 1).any())


def _fit_m3_bambi(
    df: pd.DataFrame,
    metric: str,
    *,
    bmb,
    az,
    apply_acoustic_prior: bool = True,
) -> FitResult:
    import os
    import warnings

    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

    data = df.copy()
    # Observation SE from EWSD CIs — used as inverse-variance weights / sigma floor
    data["se"] = data["se_log_obs"].fillna(0.25).clip(lower=0.05)
    data["obs_weight"] = 1.0 / (data["se"] ** 2)
    data["obs_weight"] = data["obs_weight"] / data["obs_weight"].mean()
    # Prefer measured dynamics in the likelihood design
    data_meas = data[~data["dynamic"].astype(str).str.lower().isin({"unspecified", "unknown"})]
    if len(data_meas) >= max(20, int(0.5 * len(data))):
        data = data_meas

    n_tech = int(data["technique"].nunique())
    n_dyn = int(data["dynamic"].nunique())
    n = len(data)
    use_group = _corpus_hierarchy_identifiable(data)
    group_term = " + (1|corpus_id)" if use_group else ""

    if n < 20 or n_tech == 1:
        if n_tech == 1 and n_dyn <= 1:
            formula = f"log_ratio ~ 1{group_term}"
        elif n_tech == 1:
            formula = f"log_ratio ~ 1 + C(dynamic){group_term}"
        elif n_dyn > 1:
            formula = f"log_ratio ~ 0 + technique + C(dynamic){group_term}"
        else:
            formula = f"log_ratio ~ 0 + technique{group_term}"
        draws, tune, chains = 300, 300, 2
    elif n < 60:
        formula = (
            f"log_ratio ~ 0 + technique + technique:bs(midi, df=3) + C(dynamic){group_term}"
        )
        draws, tune, chains = 400, 400, 2
    else:
        formula = (
            f"log_ratio ~ 0 + technique + technique:bs(midi, df=4) "
            f"+ technique:C(dynamic){group_term}"
        )
        draws, tune, chains = 500, 500, 2

    # Note: heteroskedastic SE²+σ² Student-t likelihood is not expressible in stock
    # Bambi formulas; se_log_obs is retained for audit / future PyMC custom likelihood.
    # Inverse-variance enters M2 WLS; here we floor residual scale awareness via transport_sd.
    weighted = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        try:
            model = bmb.Model(formula, data, family="t")
            idata = model.fit(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.9,
                progressbar=False,
            )
        except Exception:
            if group_term:
                formula = formula.replace(group_term, "")
                use_group = False
                model = bmb.Model(formula, data, family="t")
                idata = model.fit(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=0.9,
                    progressbar=False,
                )
            else:
                raise

    summary = az.summary(idata, var_names=["~mu"], filter_vars="like")
    transport_sd = (
        float(data.groupby("corpus_id")["log_ratio"].mean().std(ddof=0))
        if data["corpus_id"].nunique() > 1
        else 0.15
    )
    if not use_group:
        # Design cannot identify corpus random effects — external transport uncertainty
        transport_sd = max(transport_sd, 0.20)

    m2 = _fit_m2(df, metric, apply_acoustic_prior=apply_acoustic_prior)
    versions = {}
    try:
        import bambi as _bmb
        import pymc as _pm
        import arviz as _az

        versions = {
            "bambi": getattr(_bmb, "__version__", "?"),
            "pymc": getattr(_pm, "__version__", "?"),
            "arviz": getattr(_az, "__version__", "?"),
        }
    except Exception:
        pass

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
            "point_predict": "bambi_posterior",
            "train_dynamics": sorted(data["dynamic"].astype(str).unique().tolist()),
            "train_techniques": sorted(data["technique"].astype(str).unique().tolist()),
            "interval_type": "heuristic_predictive_from_posterior_sd",
            "observation_se_weighted": weighted,
            "corpus_group_effect": use_group,
            "package_versions": versions,
        },
        diagnostics={
            "divergences": int(idata.sample_stats["diverging"].sum().item())
            if "diverging" in idata.sample_stats
            else None,
            "transport_sd": max(transport_sd, 0.08),
            "formula": formula,
            "is_bayesian": True,
            "corpus_group_effect": use_group,
            "observation_se_weighted": weighted,
            "package_versions": versions,
            "point_predict": "bambi_posterior_with_m2_fallback",
            "note": (
                "Bayesian fixed/group-effect Student-t regression on log-ratios. "
                "Intervals combine posterior sd of mu with transport/heuristic inflate — "
                "not pure Bayesian credible intervals of Y. "
                + (
                    "Includes (1|corpus_id) because ≥1 technique spans multiple corpora."
                    if use_group
                    else "No corpus group term: technique and corpus are confounded in this bridge; "
                    "transport_sd is an external uncertainty floor."
                )
                + (
                    " Observation SE used as fit weights."
                    if weighted
                    else " Observation SE not accepted by this Bambi.fit API; se_log_obs stored only."
                )
            ),
        },
    )


def _bambi_posterior_effect(
    fit: FitResult, technique: str, dynamic: str, midi: float
) -> tuple[float, float, str] | None:
    """Posterior mean/sd of mu from Bambi; None → caller should use M2 fallback."""
    from ..acoustics import clip_log_effect

    model = fit.params.get("model")
    idata = fit.params.get("idata")
    if model is None or idata is None:
        return None
    transport = float(fit.params.get("transport_sd", 0.18))
    # Use a seen dynamic when possible (unseen levels break design matrices)
    dyns = fit.params.get("train_dynamics") or []
    dyn_use = str(dynamic)
    if dyns and dyn_use not in dyns:
        dyn_use = str(dyns[0])
    new = pd.DataFrame(
        {
            "technique": [str(technique)],
            "dynamic": [dyn_use],
            "midi": [float(midi)],
            "log_ratio": [0.0],
            "corpus_id": ["__predict__"],
            "se": [0.25],
        }
    )
    arr = None
    last_err = None
    for kind in ("response_params", "mean", "response"):
        try:
            pred = model.predict(idata, data=new, inplace=False, kind=kind)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        post = getattr(pred, "posterior", None)
        if post is None:
            continue
        for key in ("mu", "log_ratio_mean", "log_ratio"):
            if key in post:
                arr = np.asarray(post[key]).reshape(-1)
                break
        if arr is not None and arr.size:
            break
        arr = None
    if arr is None or arr.size == 0:
        if last_err is not None:
            fit.diagnostics.setdefault("bambi_predict_errors", []).append(str(last_err))
        return None
    mu = float(np.nanmean(arr))
    se = float(np.nanstd(arr))
    if not np.isfinite(mu):
        return None
    se = max(se, 0.05)
    m = (fit.params.get("models") or {}).get(technique) or {}
    midi_min, midi_max = m.get("midi_min"), m.get("midi_max")
    in_range = True
    if midi_min is not None and midi_max is not None:
        in_range = float(midi_min) <= float(midi) <= float(midi_max)
    delta, _ = clip_log_effect(mu, technique)
    se = float(np.sqrt(se**2 + transport**2))
    if not in_range:
        se = float(np.sqrt(se**2 + 0.15**2))
        return delta, se, "bambi_posterior_register_extrapolation"
    return delta, se, "bambi_posterior_mean"


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

    # M3 PyMC heteroscedastic posterior surface
    if fit.backend == "pymc_heteroscedastic_student_t":
        from .m3_pymc import predict_pymc_effect

        bayes = predict_pymc_effect(fit, technique, dynamic, midi)
        if bayes is not None:
            return bayes

    # M3 Bambi: try posterior mean before M2 fallback surface
    if (
        fit.backend == "bambi_pymc"
        and fit.params.get("point_predict") == "bambi_posterior"
    ):
        bayes = _bambi_posterior_effect(fit, technique, dynamic, midi)
        if bayes is not None:
            return bayes

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
        # PyMC constants may include MIDI slope
        if m.get("pymc_heteroscedastic"):
            from .m3_pymc import predict_pymc_effect

            bayes = predict_pymc_effect(fit, technique, dynamic, midi)
            if bayes is not None:
                return bayes
        delta, _ = clip_log_effect(float(m["mu"]), technique)
        se = float(np.sqrt(m["se"] ** 2 + transport**2))
        flag = (
            "pymc_posterior_mean"
            if m.get("pymc_heteroscedastic")
            else ("robust_constant" if in_range else "register_extrapolation")
        )
        if not in_range:
            se = float(np.sqrt(se**2 + 0.15**2))
            if not m.get("pymc_heteroscedastic"):
                flag = "register_extrapolation"
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
    calibration: Any | None = None,
) -> pd.DataFrame:
    """Apply Y_hat = Y_ord * exp(delta) with support levels and acoustic clipping."""
    from ..acoustics import clip_log_effect
    from ..dynamics import MAX_ADEQUATE_DISTANCE, map_zenodo_dynamic_to_bridge
    from ..validation.calibration import apply_conformal_halfwidth

    if max_dynamic_distance is None:
        max_dynamic_distance = MAX_ADEQUATE_DISTANCE

    # Calibrated inflate constants (defaults match legacy behaviour)
    inf_dyn = 0.25
    inf_reg = 0.12
    inf_out = 0.20
    se_scale = 1.0
    if calibration is not None and getattr(calibration, "status", None) == "ok":
        inf_dyn = float(calibration.inflate_dynamic)
        inf_reg = float(calibration.inflate_register)
        inf_out = float(calibration.inflate_outlier)
        se_scale = float(calibration.scale_factor)

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

    supported_flags = {
        "robust_constant",
        "interpolation",
        "global_factor",
        "register_dynamic",
        "dynamic_only",
        "technique_pooled",
        "bambi_posterior_mean",
        "pymc_posterior_mean",
    }

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

            measured_bridge_dyns = [
                d
                for d in (bridge_dyns or [])
                if str(d).lower() not in {"unspecified", "unknown", "nan", ""}
            ]
            unknown_bridge_dynamic = not measured_bridge_dyns

            if measured_bridge_dyns:
                effect_dyn, dyn_flag, dyn_dist = map_zenodo_dynamic_to_bridge(
                    target_dyn, measured_bridge_dyns
                )
            else:
                effect_dyn, dyn_flag, dyn_dist = "unspecified", "bridge_dynamic_unknown", np.nan

            dyn_ok = (not unknown_bridge_dynamic) and is_adequate_dynamic_pair(
                target_dyn, effect_dyn, max_dynamic_distance
            )
            if strict_dynamics and not dyn_ok:
                pass

            delta, se, model_flag = _effect_from_fit(fit, tech, effect_dyn, float(r["midi"]))
            delta, clipped = clip_log_effect(delta, tech)

            if not dyn_ok or unknown_bridge_dynamic:
                se = float(np.sqrt(se**2 + inf_dyn**2))
            if "register_extrapolation" in model_flag:
                se = float(np.sqrt(se**2 + inf_reg**2))
            se = float(np.sqrt(se**2 + transport_se_extra**2))
            se = float(se * se_scale)

            lo, hi = r.get("ci_low"), r.get("ci_high")
            if pd.notna(lo) and pd.notna(hi) and lo > 0 and hi > 0:
                se_y = (np.log(float(hi)) - np.log(float(lo))) / (2 * 1.96)
                se = float(np.sqrt(se**2 + se_y**2))

            coll = str(r.get("collection", "*"))
            cut = outlier_cut.get(coll, outlier_cut.get("*", np.inf))
            is_outlier = bool(y0 > cut)

            if unknown_bridge_dynamic:
                support_level = "extrapolated_dynamic"
            elif dyn_ok and model_flag in supported_flags:
                support_level = "supported"
            elif dyn_ok and "register_extrapolation" in model_flag:
                support_level = "extrapolated_register"
            elif not dyn_ok:
                support_level = "extrapolated_dynamic"
            else:
                support_level = "extrapolated"

            if is_outlier:
                se = float(np.sqrt(se**2 + inf_out**2))
                support_level = (
                    "supported_outlier_target"
                    if support_level == "supported"
                    else support_level + "+outlier_target"
                )

            yhat = y0 * np.exp(delta)
            half = apply_conformal_halfwidth(se, calibration)
            interval_type = str(
                fit.params.get("interval_type")
                or fit.diagnostics.get("interval_type")
                or "heuristic_predictive"
            )
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
                    "bridge_dynamic_support": "unknown"
                    if unknown_bridge_dynamic
                    else "measured",
                    "midi": r.get("midi"),
                    "note": r.get("note"),
                    "metric": fit.metric,
                    "y_ordinario": y0,
                    "target_outlier": is_outlier,
                    "log_effect": delta,
                    "factor": float(np.exp(delta)),
                    "factor_clipped": clipped,
                    "y_pred": float(yhat),
                    "y_pred_lo95": float(yhat * np.exp(-half)),
                    "y_pred_hi95": float(yhat * np.exp(half)),
                    "combined_se_log": se,
                    "interval_halfwidth_log": half,
                    "interval_type": interval_type,
                    "model_flag": model_flag,
                    "support_level": support_level,
                    "support_flag": f"{model_flag}|{dyn_flag}",
                    "estimate_class": "model_derived_synthetic",
                    "model_id": fit.model_id,
                    "backend": fit.backend,
                }
            )
    return pd.DataFrame(rows)
