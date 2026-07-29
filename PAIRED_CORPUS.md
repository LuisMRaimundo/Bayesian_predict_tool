# Genuine paired corpus (scientific priority)

## Why this matters

When ordinario and special-technique samples come from **different** collections
(folders / sessions / processing chains), the observed log-ratio is:

\[
\text{observed transform}
=
\text{technique}
+
\text{corpus}
+
\text{instrument/performer/room/mic/processing}.
\]

No Bayesian likelihood — including the new heteroscedastic Student-t — can
isolate the technique effect from that confounding. Correct labelling as
`transport_prior` is necessary but not sufficient for publication-grade claims.

## Definition of a paired corpus

For each MIDI (and preferably each dynamic), record **both**:

1. ordinario (arco normal), and  
2. the special technique (e.g. con sordino),

under the **same**:

- performer  
- instrument  
- microphone / placement  
- room  
- gain / processing / EWSD pipeline  

In this tool that means bridge rows with:

```text
special_corpus_id == ordinario_corpus_id
same_collection_pair == True
is_transport_prior == False
```

## How to use this repository

1. Build an Excel/CSV panel with paired ordinario + technique rows (same `collection` / `corpus_id`).  
2. Keep `require_same_collection=True` (default).  
3. Prefer **M1** for exploratory work; use **M3** only when Bayes extras install and you want observation-SE-aware posteriors.  
4. M3 with `require_paired_corpus_for_m3=True` (default) **refuses** transport-only bridges.

Synthetic template (not real acoustics — for CI / smoke only):

```text
data/paired_corpus_synthetic.csv
```

## Recommended experimental next step

Record one violin session covering MIDI ~55–91 at f (and ideally mf/ff) for:

- Arco normal  
- Con sordino (performance mute)  
- optionally sul ponticello / sul tasto with **annotated dynamics**

Process with the same EWSD pipeline used for Zenodo MEDIA targets, then re-run
blocked CV. Expect technique identification to become scientifically meaningful
only after such data exist.
