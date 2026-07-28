"""Clean local GUI for String Technique Transfer."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from .bridge import summarize_factors
from .config import TransferConfig
from .dynamics import ZENODO_DYNAMICS
from .io.loaders import (
    discover_zenodo_dynamic_sheets,
    find_zenodo_media_sheet,
    load_zenodo_media_ordinario,
    load_zenodo_ordinario_all,
    load_zenodo_ordinario_collection,
)
from .models.base import MODEL_CHOICES
from .pipeline import concat_panels, load_and_clean, run_transfer
from .preflight import preflight_transfer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs"


class STTApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        self.master.title("String Technique Transfer — local Bayesian tool")
        self.master.minsize(880, 640)
        self.pack(fill="both", expand=True)

        self.bridge_paths: list[str] = []
        self.target_path = tk.StringVar()
        self.zenodo_collection = tk.StringVar(value="MEDIA")  # MEDIA / ORCH / IOWA / BOTH
        self.instrument = tk.StringVar(value="Violin")
        self.model_id = tk.StringVar(value="M2_midi_gam")
        self.metric = tk.StringVar(value="EWSD_score_acoustic_balanced")
        self.same_collection = tk.BooleanVar(value=False)
        self.strict_dynamics = tk.BooleanVar(value=True)
        self.output_path = tk.StringVar(value=str(DEFAULT_OUT / "transfer_audit.xlsx"))
        self.status = tk.StringVar(value="Ready.")
        self._busy = False

        self._build()

    def _build(self) -> None:
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", pady=(0, 8))
        ttk.Label(hdr, text="String Technique Transfer", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            hdr,
            text="Transport special-technique effects onto target ordinario profiles (violin / viola / cello / bass).",
            foreground="#444",
        ).pack(anchor="w")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        left = ttk.LabelFrame(body, text="Inputs", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Instrument
        row = ttk.Frame(left)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Instrument", width=16).pack(side="left")
        ttk.Combobox(
            row,
            textvariable=self.instrument,
            values=["Violin", "Viola", "Cello", "Double Bass", "Other"],
            width=28,
        ).pack(side="left", fill="x", expand=True)

        # Bridge files
        bf = ttk.LabelFrame(left, text="Bridge panel (ordinario + techniques)", padding=8)
        bf.pack(fill="both", expand=True, pady=6)
        self.bridge_list = tk.Listbox(bf, height=7)
        self.bridge_list.pack(fill="both", expand=True)
        btns = ttk.Frame(bf)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Add file…", command=self.add_bridge).pack(side="left")
        ttk.Button(btns, text="Remove", command=self.remove_bridge).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear", command=self.clear_bridge).pack(side="left")

        # Target
        tf = ttk.LabelFrame(left, text="Target ordinario", padding=8)
        tf.pack(fill="x", pady=6)
        row = ttk.Frame(tf)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.target_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self.browse_target).pack(side="left", padx=4)
        row2 = ttk.Frame(tf)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="Zenodo collection").pack(side="left")
        ttk.Combobox(
            row2,
            textvariable=self.zenodo_collection,
            values=["MEDIA", "ORCH", "IOWA", "BOTH"],
            width=12,
            state="readonly",
        ).pack(side="left", padx=6)
        ttk.Label(
            row2,
            text=f"(always loads {', '.join(ZENODO_DYNAMICS)} — nearest dynamic matching)",
            foreground="#666",
        ).pack(side="left")

        # Model
        mf = ttk.LabelFrame(left, text="Model", padding=8)
        mf.pack(fill="x", pady=6)
        row = ttk.Frame(mf)
        row.pack(fill="x")
        ttk.Label(row, text="Model", width=16).pack(side="left")
        ttk.Combobox(row, textvariable=self.model_id, values=list(MODEL_CHOICES), width=32, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        row = ttk.Frame(mf)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Metric", width=16).pack(side="left")
        ttk.Entry(row, textvariable=self.metric).pack(side="left", fill="x", expand=True)
        ttk.Checkbutton(
            mf,
            text="Require same-collection pairing (strict bridge)",
            variable=self.same_collection,
        ).pack(anchor="w", pady=4)
        ttk.Checkbutton(
            mf,
            text="Strict dynamics (only adequate matches, e.g. f→ff; exclude pp/mf if unsupported)",
            variable=self.strict_dynamics,
        ).pack(anchor="w", pady=2)

        # Output
        of = ttk.LabelFrame(left, text="Output Excel audit", padding=8)
        of.pack(fill="x", pady=6)
        row = ttk.Frame(of)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.output_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self.browse_output).pack(side="left", padx=4)

        # Actions
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=8)
        self.preflight_btn = ttk.Button(actions, text="Preflight", command=self.preflight)
        self.preflight_btn.pack(side="left")
        self.run_btn = ttk.Button(actions, text="Fit & predict", command=self.run)
        self.run_btn.pack(side="left", padx=8)
        ttk.Button(actions, text="Open output folder", command=self.open_out).pack(side="left")

        # Right: log
        right = ttk.LabelFrame(body, text="Log", padding=8)
        right.pack(side="left", fill="both", expand=True)
        self.log = tk.Text(right, wrap="word", height=30, font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
        self._log(
            "Local tool — not a GitHub project.\n"
            "1) Add bridge files (ordinario + special techniques).\n"
            "2) Choose Zenodo workbook + collection (ORCH/IOWA/BOTH).\n"
            "3) Click Preflight (recommended), keep Strict dynamics ON.\n"
            "4) Prefer M2; Fit & predict.\n"
            "5) USE: [your .xlsx file] → sheet Predictions_supported → column y_pred "
            "(yellow highlight in Excel).\n"
        )

        foot = ttk.Frame(self)
        foot.pack(fill="x", pady=(8, 0))
        ttk.Label(foot, textvariable=self.status).pack(anchor="w")

    def _log(self, msg: str) -> None:
        self.log.insert("end", msg.rstrip() + "\n")
        self.log.see("end")

    def add_bridge(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Bridge panel files",
            filetypes=[("Data", "*.csv *.xlsx *.xlsm"), ("All", "*.*")],
        )
        for p in paths:
            if p not in self.bridge_paths:
                self.bridge_paths.append(p)
                self.bridge_list.insert("end", p)

    def remove_bridge(self) -> None:
        sel = list(self.bridge_list.curselection())
        for i in reversed(sel):
            self.bridge_list.delete(i)
            del self.bridge_paths[i]

    def clear_bridge(self) -> None:
        self.bridge_list.delete(0, "end")
        self.bridge_paths.clear()

    def browse_target(self) -> None:
        p = filedialog.askopenfilename(
            title="Target ordinario",
            filetypes=[("Data", "*.csv *.xlsx *.xlsm"), ("All", "*.*")],
        )
        if p:
            self.target_path.set(p)

    def browse_output(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Output Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile="transfer_audit.xlsx",
        )
        if p:
            self.output_path.set(p)

    def open_out(self) -> None:
        folder = Path(self.output_path.get()).expanduser().resolve().parent
        folder.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception:
            self._log(str(folder))

    def _cfg(self) -> TransferConfig:
        return TransferConfig(
            metric=self.metric.get().strip(),
            model_id=self.model_id.get(),
            require_same_collection=self.same_collection.get(),
            strict_dynamics=self.strict_dynamics.get(),
            run_blocked_cv=True,
        )

    def _load_bridge_and_target(self, metric: str):
        frames = []
        for p in self.bridge_paths:
            self._ui_log(f"Loading bridge: {p}")
            clean, dups, audit = load_and_clean(p, metric=metric)
            if (clean["instrument"] == "unknown").all():
                clean["instrument"] = self.instrument.get()
                clean["corpus_id"] = clean["instrument"] + "|" + clean["collection"].astype(str)
            frames.append(clean)
            techs = sorted(clean["technique"].dropna().unique().tolist())
            dyns = sorted(clean["dynamic"].dropna().unique().tolist())
            n_ord = int(clean["is_ordinario"].sum()) if "is_ordinario" in clean.columns else 0
            self._ui_log(
                f"  rows={audit['n_rows']} unique={audit['n_unique_keys']} "
                f"dup_rows={audit['n_duplicate_rows']}"
            )
            self._ui_log(f"  inferred technique={techs} dynamic={dyns} ordinario_rows={n_ord}")
            if len(dups):
                self._ui_log(f"  removed/flagged duplicate key rows: {len(dups)}")
        bridge_panel = concat_panels(frames)

        tpath = Path(self.target_path.get())
        self._ui_log(f"Loading target: {tpath}")
        use_zenodo = False
        media_sheet = None
        if tpath.suffix.lower() in {".xlsx", ".xlsm"}:
            try:
                groups = discover_zenodo_dynamic_sheets(tpath, instrument=self.instrument.get())
                media_sheet = find_zenodo_media_sheet(tpath, instrument=self.instrument.get())
                use_zenodo = bool(groups) or bool(media_sheet)
            except Exception:
                use_zenodo = "zenodo" in tpath.name.lower()
                media_sheet = find_zenodo_media_sheet(tpath, instrument=self.instrument.get())
        if use_zenodo:
            coll = self.zenodo_collection.get().strip().upper()
            if coll in {"MEDIA", "BOTH"} and media_sheet:
                target = load_zenodo_media_ordinario(
                    tpath, instrument=self.instrument.get(), metric=metric
                )
                src_cols = sorted(target["source_column"].dropna().astype(str).unique())
                self._ui_log(
                    f"Zenodo MEDIA sheet: {media_sheet} "
                    f"(columns {src_cols}; ff = Excel column O / Media ff)"
                )
            elif coll == "BOTH":
                target = load_zenodo_ordinario_all(
                    tpath, instrument=self.instrument.get(), metric=metric
                )
                self._ui_log(f"Zenodo collections loaded: {sorted(target['collection'].unique())}")
            elif coll == "MEDIA" and not media_sheet:
                raise ValueError(
                    f"No *_Media sheet in {tpath.name}. Use ORCH/IOWA or add Violin_Media."
                )
            else:
                target = load_zenodo_ordinario_collection(
                    tpath,
                    collection=coll,
                    instrument=self.instrument.get(),
                    metric=metric,
                )
                self._ui_log(f"Zenodo collection: {coll} (Combined density metric sheets)")
            dyns = sorted(target["dynamic"].dropna().astype(str).unique().tolist())
            self._ui_log(
                f"  target rows={len(target)} dynamics={dyns} "
                f"(expected triad {list(ZENODO_DYNAMICS)})"
            )
        else:
            target, _, _ = load_and_clean(tpath, metric=metric)
            if (target["instrument"] == "unknown").all():
                target["instrument"] = self.instrument.get()
            if "is_ordinario" in target.columns and target["is_ordinario"].any():
                target = target.loc[target["is_ordinario"]].copy()
            else:
                target["technique"] = "ordinario"
                target["is_ordinario"] = True
        return bridge_panel, target

    def preflight(self) -> None:
        if self._busy:
            return
        if not self.bridge_paths or not self.target_path.get():
            messagebox.showerror("Missing inputs", "Add bridge file(s) and a target ordinario file.")
            return
        self._busy = True
        self.preflight_btn.state(["disabled"])
        self.run_btn.state(["disabled"])
        self.status.set("Preflight…")
        threading.Thread(target=self._preflight_worker, daemon=True).start()

    def _preflight_worker(self) -> None:
        try:
            metric = self.metric.get().strip()
            bridge_panel, target = self._load_bridge_and_target(metric)
            pf = preflight_transfer(bridge_panel, target, self._cfg())
            self._ui_log("\n=== PREFLIGHT ===")
            self._ui_log(pf.as_dataframe().to_string(index=False))
            self._ui_status("Preflight PASS" if pf.ok else "Preflight FAIL")
            msg = "Preflight PASS — safe to Fit & predict." if pf.ok else "Preflight FAIL — see log."
            self.master.after(
                0,
                lambda m=msg, ok=pf.ok: messagebox.showinfo("Preflight", m)
                if ok
                else messagebox.showerror("Preflight", m),
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            err_msg = str(exc)
            self._ui_log("ERROR:\n" + tb)
            self._ui_status("Failed.")
            self.master.after(0, lambda m=err_msg: messagebox.showerror("Error", m))
        finally:
            self._busy = False
            self.master.after(0, lambda: self.preflight_btn.state(["!disabled"]))
            self.master.after(0, lambda: self.run_btn.state(["!disabled"]))

    def run(self) -> None:
        if self._busy:
            return
        if not self.bridge_paths:
            messagebox.showerror("Missing bridge", "Add at least one bridge panel file.")
            return
        if not self.target_path.get():
            messagebox.showerror("Missing target", "Choose a target ordinario file.")
            return
        self._busy = True
        self.run_btn.state(["disabled"])
        self.preflight_btn.state(["disabled"])
        self.status.set("Running…")
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self) -> None:
        try:
            metric = self.metric.get().strip()
            bridge_panel, target = self._load_bridge_and_target(metric)
            out = Path(self.output_path.get()).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            fit, bridge, preds, out_path, pf, cv = run_transfer(
                bridge_panel,
                target,
                config=self._cfg(),
                output_xlsx=out,
            )
            factors = summarize_factors(bridge)
            self._ui_log("\n=== PREFLIGHT ===")
            self._ui_log(pf.as_dataframe().to_string(index=False))
            self._ui_log("\n" + fit.summary_text())
            self._ui_log("\nFactor summary (median):")
            self._ui_log(factors.to_string(index=False))
            if len(cv):
                self._ui_log("\nBlocked CV:")
                self._ui_log(cv.to_string(index=False))
            if len(preds) and "support_level" in preds.columns:
                sup = preds[preds["support_level"].isin(["supported", "supported_outlier_target"])]
                self._ui_log(
                    f"\nSupport: supported={len(sup)} / all={len(preds)} "
                    f"(strict_dynamics={self.strict_dynamics.get()})"
                )
                tab = (
                    preds.groupby(
                        ["dynamic", "bridge_dynamic_used", "support_level", "dynamic_adequate"],
                        dropna=False,
                    )
                    .size()
                    .reset_index(name="n")
                )
                self._ui_log(tab.to_string(index=False))
                if len(sup):
                    self._ui_log(
                        "\nSupported medians by dynamic:\n"
                        + sup.groupby("dynamic")[["y_ordinario", "y_pred", "factor"]]
                        .median()
                        .to_string()
                    )
            self._ui_log(f"\nExcel audit: {out_path}")
            self._ui_log("=== USE THESE (mimic special technique on IOWA/ORCHIDEA) ===")
            self._ui_log(f"1. FILE NAME : {Path(out_path).name}")
            self._ui_log("2. PAGE NAME : Predictions_supported")
            self._ui_log("3. COLUMN NAME: y_pred  (yellow highlight)")
            self._ui_status("Done.")
            n_sup = (
                int(preds["support_level"].isin(["supported", "supported_outlier_target"]).sum())
                if len(preds) and "support_level" in preds.columns
                else 0
            )
            done_msg = (
                f"Wrote\n{out_path}\n\n"
                f"Supported rows: {n_sup} / {len(preds)}\n\n"
                f"1. FILE: {Path(out_path).name}\n"
                f"2. PAGE: Predictions_supported\n"
                f"3. COLUMN: y_pred"
            )
            self.master.after(0, lambda m=done_msg: messagebox.showinfo("Done", m))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            err_msg = str(exc)
            self._ui_log("ERROR:\n" + tb)
            self._ui_status("Failed.")
            self.master.after(0, lambda m=err_msg: messagebox.showerror("Error", m))
        finally:
            self._busy = False
            self.master.after(0, lambda: self.run_btn.state(["!disabled"]))
            self.master.after(0, lambda: self.preflight_btn.state(["!disabled"]))

    def _ui_log(self, msg: str) -> None:
        self.master.after(0, lambda: self._log(msg))

    def _ui_status(self, msg: str) -> None:
        self.master.after(0, lambda: self.status.set(msg))


def main() -> None:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:
        pass
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    STTApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
