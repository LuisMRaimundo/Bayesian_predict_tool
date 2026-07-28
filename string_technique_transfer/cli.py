"""Command-line entry (local)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .bridge import summarize_factors
from .io.loaders import (
    discover_zenodo_dynamic_sheets,
    find_zenodo_media_sheet,
    load_zenodo_media_ordinario,
    load_zenodo_ordinario_all,
    load_zenodo_ordinario_collection,
)
from .pipeline import concat_panels, load_and_clean, run_transfer
from .preflight import preflight_transfer
from .config import TransferConfig


def _load_inputs(args):
    frames = []
    for b in args.bridge:
        clean, _, audit = load_and_clean(b, metric=args.metric)
        if (clean["instrument"] == "unknown").all():
            clean["instrument"] = args.instrument
            clean["corpus_id"] = clean["instrument"] + "|" + clean["collection"].astype(str)
        print(f"bridge {b}: {audit}")
        frames.append(clean)
    bridge_panel = concat_panels(frames)

    tpath = Path(args.target)
    groups = {}
    media_sheet = find_zenodo_media_sheet(tpath, instrument=args.instrument)
    try:
        groups = discover_zenodo_dynamic_sheets(tpath, instrument=args.instrument)
    except Exception:
        groups = {}
    if groups or media_sheet:
        coll = args.zenodo_collection.strip().upper()
        if coll in {"MEDIA", "BOTH"} and media_sheet:
            target = load_zenodo_media_ordinario(
                tpath, instrument=args.instrument, metric=args.metric
            )
            print(
                f"zenodo MEDIA: sheet={media_sheet} "
                f"columns={sorted(target['source_column'].unique())} "
                f"(ff = column O / Media ff) rows={len(target)}"
            )
        elif coll == "BOTH":
            target = load_zenodo_ordinario_all(
                tpath, instrument=args.instrument, metric=args.metric
            )
        elif coll == "MEDIA":
            raise ValueError(f"No *_Media sheet in {tpath.name}")
        else:
            target = load_zenodo_ordinario_collection(
                tpath,
                collection=coll,
                instrument=args.instrument,
                metric=args.metric,
            )
        print(
            f"zenodo target: collections={sorted(target['collection'].unique())} "
            f"dynamics={sorted(target['dynamic'].unique())} rows={len(target)}"
        )
    else:
        target, _, _ = load_and_clean(tpath, metric=args.metric)
        if "is_ordinario" in target.columns and target["is_ordinario"].any():
            target = target.loc[target["is_ordinario"]].copy()
    return bridge_panel, target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="String Technique Transfer (local)")
    p.add_argument("--bridge", nargs="+", required=True, help="Bridge panel CSV/XLSX files")
    p.add_argument("--target", required=True, help="Target ordinario CSV/XLSX")
    p.add_argument(
        "--zenodo-collection",
        default="MEDIA",
        help="Zenodo source: MEDIA (Violin_Media!O/N/M) | ORCH | IOWA | BOTH",
    )
    p.add_argument("--instrument", default="Violin")
    p.add_argument("--model", default="M2_midi_gam")
    p.add_argument("--metric", default="EWSD_score_acoustic_balanced")
    p.add_argument("--same-collection", action="store_true")
    p.add_argument(
        "--no-strict-dynamics",
        action="store_true",
        help="Allow inadequate dynamic matches (e.g. f->pp) as extrapolated rows",
    )
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--no-cv", action="store_true")
    p.add_argument("--out", default=str(Path("outputs") / "transfer_audit.xlsx"))
    args = p.parse_args(argv)

    bridge_panel, target = _load_inputs(args)
    cfg = TransferConfig(
        metric=args.metric,
        model_id=args.model,
        require_same_collection=args.same_collection,
        strict_dynamics=not args.no_strict_dynamics,
        run_blocked_cv=not args.no_cv,
    )
    run_meta = {
        "kind": "preflight" if args.preflight_only else "transfer",
        "bridge_paths": list(args.bridge),
        "target_path": args.target,
        "instrument": args.instrument,
        "zenodo_collection": args.zenodo_collection,
    }
    pf = preflight_transfer(bridge_panel, target, cfg)
    print(pf.as_dataframe().to_string(index=False))
    if args.preflight_only:
        from .run_history import finalize_run, log_operation, start_run

        rec = start_run(
            kind="preflight",
            bridge_paths=list(args.bridge),
            target_path=args.target,
            config=cfg.to_dict(),
            instrument=args.instrument,
            zenodo_collection=args.zenodo_collection,
        )
        log_operation(rec, "preflight", {"ok": pf.ok})
        report = finalize_run(
            rec,
            status="ok" if pf.ok else "preflight_fail",
            bridge_panel=bridge_panel,
            target=target,
            preflight_df=pf.as_dataframe(),
            errors=pf.errors,
            warnings=pf.warnings,
        )
        print(f"Run history: {report}")
        return 0 if pf.ok else 2
    if not pf.ok:
        print("Preflight FAILED — aborting.")
        return 2

    fit, bridge, preds, out, _pf, cv = run_transfer(
        bridge_panel,
        target,
        config=cfg,
        output_xlsx=args.out,
        skip_preflight=True,
        run_meta=run_meta,
    )
    print(fit.summary_text())
    print(summarize_factors(bridge).to_string(index=False))
    if len(cv):
        print("Blocked CV:")
        print(cv.to_string(index=False))
    if len(preds) and "support_level" in preds.columns:
        tab = (
            preds.groupby(["dynamic", "bridge_dynamic_used", "support_level", "dynamic_adequate"])
            .size()
            .reset_index(name="n")
        )
        print(tab.to_string(index=False))
        n_sup = int(preds["support_level"].isin(["supported", "supported_outlier_target"]).sum())
        print(f"supported={n_sup} / all={len(preds)}")
    print(f"wrote {out}")
    if fit.diagnostics.get("run_history_report"):
        print(f"Run history: {fit.diagnostics['run_history_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
