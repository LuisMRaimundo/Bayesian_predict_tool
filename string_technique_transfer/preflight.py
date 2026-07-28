"""Pre-flight checks before fitting/transfer."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .bridge import build_log_ratios
from .config import TransferConfig
from .dynamics import supported_zenodo_dynamics_for_bridge
from .schema import is_ordinario


@dataclass
class PreflightResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def as_dataframe(self) -> pd.DataFrame:
        rows = [{"level": "INFO", "message": f"{k}={v}"} for k, v in self.summary.items()]
        rows += [{"level": "ERROR", "message": e} for e in self.errors]
        rows += [{"level": "WARNING", "message": w} for w in self.warnings]
        rows.append({"level": "STATUS", "message": "PASS" if self.ok else "FAIL"})
        return pd.DataFrame(rows)


def preflight_transfer(
    bridge_panel: pd.DataFrame,
    target_ordinario: pd.DataFrame,
    config: TransferConfig | None = None,
) -> PreflightResult:
    cfg = config or TransferConfig()
    cfg.validate()
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict = {}

    if bridge_panel is None or len(bridge_panel) == 0:
        errors.append("Bridge panel is empty.")
    if target_ordinario is None or len(target_ordinario) == 0:
        errors.append("Target ordinario is empty.")
    if errors:
        return PreflightResult(False, errors, warnings, summary)

    n_ord = int(bridge_panel["technique"].map(is_ordinario).sum()) if "technique" in bridge_panel else 0
    n_tech = int((~bridge_panel["technique"].map(is_ordinario)).sum()) if "technique" in bridge_panel else 0
    summary["bridge_rows"] = int(len(bridge_panel))
    summary["bridge_ordinario_rows"] = n_ord
    summary["bridge_special_rows"] = n_tech
    summary["target_rows"] = int(len(target_ordinario))
    summary["target_dynamics"] = ", ".join(
        sorted(target_ordinario["dynamic"].dropna().astype(str).unique())
    )
    summary["model_id"] = cfg.model_id
    summary["strict_dynamics"] = cfg.strict_dynamics

    if n_ord == 0:
        errors.append("Bridge has no ordinario rows.")
    if n_tech == 0:
        errors.append("Bridge has no special-technique rows.")

    try:
        bridge = build_log_ratios(
            bridge_panel,
            metric=cfg.metric,
            require_same_collection=cfg.require_same_collection,
            max_dynamic_distance=cfg.max_dynamic_distance,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Cannot build bridge log-ratios: {exc}")
        return PreflightResult(False, errors, warnings, summary)

    summary["bridge_pairs"] = int(len(bridge))
    summary["bridge_techniques"] = ", ".join(sorted(bridge["technique"].astype(str).unique()))
    summary["bridge_dynamics"] = ", ".join(sorted(bridge["dynamic"].astype(str).unique()))
    summary["bridge_midi_min"] = float(bridge["midi"].min())
    summary["bridge_midi_max"] = float(bridge["midi"].max())
    if "is_transport_prior" in bridge:
        summary["transport_prior_fraction"] = round(float(bridge["is_transport_prior"].mean()), 3)

    if len(bridge) < cfg.min_bridge_pairs:
        warnings.append(
            f"Only {len(bridge)} bridge pairs (< {cfg.min_bridge_pairs}). "
            "Estimates will be unstable; prefer a richer paired bridge."
        )

    supported_by_tech = {}
    for tech, g in bridge.groupby("technique"):
        zsup = supported_zenodo_dynamics_for_bridge(
            g["dynamic"].unique(), max_distance=cfg.max_dynamic_distance
        )
        supported_by_tech[str(tech)] = zsup
        summary[f"supported_zenodo[{tech}]"] = ", ".join(zsup) if zsup else "(none)"
        if not zsup:
            warnings.append(
                f"Technique '{tech}' has no adequately supported Zenodo dynamics "
                f"from bridge dynamics={sorted(g['dynamic'].astype(str).unique())}."
            )

    tgt_dyns = set(target_ordinario["dynamic"].dropna().astype(str).str.lower())
    any_support = any(len(v) for v in supported_by_tech.values())
    if cfg.strict_dynamics and not any_support:
        errors.append(
            "Strict dynamics: no Zenodo target dynamics are adequately supported by the bridge."
        )
    elif cfg.strict_dynamics:
        # estimate how many target rows would be supported dynamically
        n_possible = 0
        for tech, zsup in supported_by_tech.items():
            n_possible += int(target_ordinario["dynamic"].astype(str).str.lower().isin(zsup).sum())
        summary["approx_supported_target_rows"] = n_possible
        if n_possible < cfg.min_supported_predictions:
            errors.append(
                f"Strict dynamics would yield < {cfg.min_supported_predictions} supported target rows."
            )
        unused = sorted(tgt_dyns - set().union(*[set(v) for v in supported_by_tech.values()]))
        if unused:
            warnings.append(
                "Target dynamics without adequate bridge support (will be extrapolated only): "
                + ", ".join(unused)
            )

    if summary.get("transport_prior_fraction", 0) > 0.5:
        warnings.append(
            "Most bridge pairs are cross-collection transport priors; "
            "do not interpret effects as causal technique-only differences."
        )

    if bridge["technique"].nunique() == 1 and bridge["dynamic"].nunique() == 1:
        warnings.append(
            "Single technique and single dynamic in bridge: M3 full Bayes adds little; M2 robust constant is preferred."
        )

    ok = len(errors) == 0
    return PreflightResult(ok=ok, errors=errors, warnings=warnings, summary=summary)
