"""M3 heteroscedastic Student-t likelihood in PyMC (observation SE used).

Likelihood for bridge log-ratios r_i:

    r_i ~ StudentT(ν, μ_i, sqrt(SE_log,i² + σ²))

where SE_log,i comes from EWSD confidence intervals (``se_log_obs``).
Mean structure is technique intercepts + optional MIDI slope + optional
dynamic offsets. A corpus group effect is added only when identifiable
(≥1 technique spans ≥2 corpora).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import FitResult


def _import_pymc():
    try:
        import arviz as az
        import pymc as pm
        import pytensor.tensor as pt

        return pm, az, pt
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "PyMC/ArviZ required for heteroscedastic M3. "
            "Install: pip install -r requirements-bayes.txt"
        ) from exc


def _corpus_hierarchy_identifiable(data: pd.DataFrame) -> bool:
    if "corpus_id" not in data.columns:
        return False
    return bool((data.groupby("technique")["corpus_id"].nunique() > 1).any())


def fit_m3_heteroscedastic(
    df: pd.DataFrame,
    metric: str,
    *,
    apply_acoustic_prior: bool = True,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    target_accept: float = 0.9,
    m2_fallback_fitter=None,
) -> FitResult:
    """Fit observation-SE-aware Student-t model; return FitResult for predict_transfer."""
    pm, az, pt = _import_pymc()
    import os

    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

    data = df.copy()
    data = data.dropna(subset=["log_ratio", "midi", "technique"]).copy()
    if "se_log_obs" not in data.columns:
        data["se_log_obs"] = np.nan
    data["se"] = data["se_log_obs"].astype(float)
    # Floor for missing CI; never discard SE when present
    data["se"] = data["se"].where(np.isfinite(data["se"]) & (data["se"] > 0), 0.20)
    data["se"] = data["se"].clip(lower=0.03, upper=1.5)

    # Prefer measured dynamics for design matrix; keep SE on those rows
    meas = data[~data["dynamic"].astype(str).str.lower().isin({"unspecified", "unknown"})]
    if len(meas) >= max(12, int(0.4 * len(data))):
        data = meas

    tech_levels = sorted(data["technique"].astype(str).unique().tolist())
    dyn_levels = sorted(data["dynamic"].astype(str).unique().tolist())
    tech_idx = data["technique"].astype(str).map({t: i for i, t in enumerate(tech_levels)}).to_numpy()
    dyn_idx = data["dynamic"].astype(str).map({d: i for i, d in enumerate(dyn_levels)}).to_numpy()
    midi = data["midi"].to_numpy(dtype=float)
    midi_c = (midi - float(np.mean(midi))) / max(float(np.std(midi)), 1.0)
    y = data["log_ratio"].to_numpy(dtype=float)
    se = data["se"].to_numpy(dtype=float)
    n = len(data)
    n_tech = len(tech_levels)
    n_dyn = len(dyn_levels)

    use_group = _corpus_hierarchy_identifiable(data)
    corpus_levels: list[str] = []
    corpus_idx = np.zeros(n, dtype=int)
    if use_group:
        corpus_levels = sorted(data["corpus_id"].astype(str).unique().tolist())
        corpus_idx = (
            data["corpus_id"].astype(str).map({c: i for i, c in enumerate(corpus_levels)}).to_numpy()
        )

    # Acoustic prior centers for technique intercepts (weak)
    prior_mu = np.zeros(n_tech)
    if apply_acoustic_prior:
        from ..acoustics import TECHNIQUE_PRIOR

        for i, tech in enumerate(tech_levels):
            conf = TECHNIQUE_PRIOR.get(tech)
            if conf is not None:
                prior_mu[i] = float(conf["prior_log"])

    coords = {
        "technique": tech_levels,
        "dynamic": dyn_levels,
        "obs": np.arange(n),
    }
    if use_group:
        coords["corpus"] = corpus_levels

    with pm.Model(coords=coords) as model:
        # Technique intercepts ~ N(prior_mu, 0.35)
        alpha = pm.Normal("alpha", mu=prior_mu, sigma=0.35, dims="technique")
        # Mild MIDI slope per technique
        beta = pm.Normal("beta_midi", mu=0.0, sigma=0.15, dims="technique")
        # Dynamic offsets (sum-to-zero via soft prior)
        if n_dyn > 1:
            gamma_raw = pm.Normal("gamma_dyn_raw", mu=0.0, sigma=0.15, dims="dynamic")
            gamma = pm.Deterministic("gamma_dyn", gamma_raw - pt.mean(gamma_raw), dims="dynamic")
            mu = alpha[tech_idx] + beta[tech_idx] * midi_c + gamma[dyn_idx]
        else:
            mu = alpha[tech_idx] + beta[tech_idx] * midi_c

        if use_group:
            tau_u = pm.HalfNormal("tau_corpus", sigma=0.15)
            u = pm.Normal("u_corpus", mu=0.0, sigma=tau_u, dims="corpus")
            mu = mu + u[corpus_idx]

        sigma = pm.HalfNormal("sigma_resid", sigma=0.25)
        nu = pm.Gamma("nu", alpha=3.0, beta=0.1)
        # Heteroscedastic scale: sqrt(SE_i^2 + σ^2)
        scale = pt.sqrt(se**2 + sigma**2)
        pm.StudentT("r", nu=nu, mu=mu, sigma=scale, observed=y, dims="obs")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            progressbar=False,
            random_seed=42,
        )

    # Posterior technique effects at mean MIDI, reference dynamic
    alpha_post = idata.posterior["alpha"].mean(dim=("chain", "draw")).values
    beta_post = idata.posterior["beta_midi"].mean(dim=("chain", "draw")).values
    alpha_sd = idata.posterior["alpha"].std(dim=("chain", "draw")).values
    sigma_post = float(idata.posterior["sigma_resid"].mean().values)
    nu_post = float(idata.posterior["nu"].mean().values)

    effect_rows = []
    models: dict[str, Any] = {}
    midi_min = float(data["midi"].min())
    midi_max = float(data["midi"].max())
    for i, tech in enumerate(tech_levels):
        mu_t = float(alpha_post[i])  # at centered midi = 0
        se_t = float(max(alpha_sd[i], 0.05))
        models[tech] = {
            "type": "constant",
            "mu": mu_t,
            "se": se_t,
            "beta_midi": float(beta_post[i]),
            "midi_center": float(np.mean(midi)),
            "midi_scale": max(float(np.std(midi)), 1.0),
            "midi_min": midi_min,
            "midi_max": midi_max,
            "pymc_heteroscedastic": True,
        }
        effect_rows.append(
            {
                "technique": tech,
                "dynamic": "all",
                "register": "all",
                "log_effect": mu_t,
                "se": se_t,
                "n": int((data["technique"] == tech).sum()),
                "fit_type": "pymc_student_t_heteroscedastic",
            }
        )

    # Optional M2 surface for out-of-support prediction fallback
    m2_models = {}
    transport_sd = 0.18
    if m2_fallback_fitter is not None:
        m2 = m2_fallback_fitter(df, metric, apply_acoustic_prior=apply_acoustic_prior)
        m2_models = m2.params.get("models", {})
        transport_sd = float(m2.params.get("transport_sd", 0.18))

    # Prefer PyMC constants; keep M2 GAM only as named fallback
    for tech, m in models.items():
        if tech in m2_models and m2_models[tech].get("type") == "gam":
            m["m2_gam_fallback"] = m2_models[tech]

    if "is_transport_prior" in data.columns and float(data["is_transport_prior"].mean()) > 0.5:
        transport_sd = max(transport_sd, 0.22)

    try:
        import pymc as _pm
        import arviz as _az

        versions = {
            "pymc": getattr(_pm, "__version__", "?"),
            "arviz": getattr(_az, "__version__", "?"),
        }
    except Exception:
        versions = {}

    summary = az.summary(idata, var_names=["alpha", "beta_midi", "sigma_resid", "nu"])
    if use_group:
        summary = az.summary(
            idata, var_names=["alpha", "beta_midi", "sigma_resid", "nu", "tau_corpus"]
        )

    return FitResult(
        model_id="M3_hierarchical_bayes",
        backend="pymc_heteroscedastic_student_t",
        metric=metric,
        bridge_n=n,
        effects=summary.reset_index().rename(columns={"index": "parameter"}),
        params={
            "idata": idata,
            "pymc_model": model,
            "models": models,
            "transport_sd": transport_sd,
            "transport_sd_source": "external_proxy_or_m2",
            "midi_range": {t: (midi_min, midi_max) for t in tech_levels},
            "point_predict": "pymc_posterior_technique",
            "train_dynamics": dyn_levels,
            "train_techniques": tech_levels,
            "interval_type": "posterior_sd_plus_transport_proxy",
            "observation_se_in_likelihood": True,
            "corpus_group_effect": use_group,
            "package_versions": versions,
            "sigma_resid": sigma_post,
            "nu": nu_post,
            "likelihood": "StudentT(nu, mu, sqrt(se_log_obs^2 + sigma^2))",
        },
        diagnostics={
            "is_bayesian": True,
            "observation_se_in_likelihood": True,
            "corpus_group_effect": use_group,
            "divergences": int(idata.sample_stats["diverging"].sum().item())
            if "diverging" in idata.sample_stats
            else None,
            "sigma_resid": sigma_post,
            "nu": nu_post,
            "package_versions": versions,
            "likelihood": "r ~ StudentT(nu, mu, sqrt(SE^2 + sigma^2))",
            "note": (
                "Heteroscedastic Student-t: each bridge row uses its se_log_obs in the scale. "
                + (
                    "Corpus random effect included (technique spans multiple corpora)."
                    if use_group
                    else "No corpus random effect: not identifiable when technique≡corpus."
                )
            ),
        },
    )


def predict_pymc_effect(
    fit: FitResult, technique: str, dynamic: str, midi: float
) -> tuple[float, float, str] | None:
    """Point effect from PyMC posterior technique surface."""
    from ..acoustics import clip_log_effect

    if fit.backend != "pymc_heteroscedastic_student_t":
        return None
    models = fit.params.get("models") or {}
    m = models.get(technique)
    if not m or not m.get("pymc_heteroscedastic"):
        return None
    center = float(m.get("midi_center", 70.0))
    scale = float(m.get("midi_scale", 1.0))
    midi_c = (float(midi) - center) / max(scale, 1e-6)
    mu = float(m["mu"]) + float(m.get("beta_midi", 0.0)) * midi_c
    # Optional dynamic offset from posterior if stored in idata
    idata = fit.params.get("idata")
    dyns = fit.params.get("train_dynamics") or []
    if idata is not None and "gamma_dyn" in getattr(idata, "posterior", {}) and dyns:
        if str(dynamic) in dyns:
            g = idata.posterior["gamma_dyn"].mean(dim=("chain", "draw"))
            # xarray with dynamic coord
            try:
                mu = mu + float(g.sel(dynamic=str(dynamic)).values)
            except Exception:
                pass
    se = float(m.get("se", 0.15))
    transport = float(fit.params.get("transport_sd", 0.18))
    sigma = float(fit.params.get("sigma_resid", 0.2))
    se = float(np.sqrt(se**2 + transport**2 + 0.25 * sigma**2))
    midi_min, midi_max = m.get("midi_min"), m.get("midi_max")
    in_range = True
    if midi_min is not None and midi_max is not None:
        in_range = float(midi_min) <= float(midi) <= float(midi_max)
    delta, _ = clip_log_effect(mu, technique)
    if not in_range:
        se = float(np.sqrt(se**2 + 0.15**2))
        return delta, se, "pymc_posterior_register_extrapolation"
    return delta, se, "pymc_posterior_mean"
