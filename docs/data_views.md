# Data Views

## Purpose
Expose multiple **label views** so models can train/evaluate in binary or multi-class **without reprocessing**. For how raw sources are standardized, see `docs/load_data.md`. Romanization options are in `docs/romanization_notes.md`.

**Sources**
- Inspection‑AI Japanese Toxic Dataset — vote counts per fine label + category vote columns.  
- LLM‑jp Toxicity Dataset v2 — single tri‑state label + per‑type category flags.

## Standardized schema (per row)
We use the unified schema described in `docs/load_data.md`:  
`id, text_native, label_text_fine, label_int_fine, label_text_coarse, label_int_coarse, label_conf, low_confidence, categories, source, split, orig_label, orig_meta`

- `source ∈ {inspection-ai, llm-jp}`
- `categories` is a JSON list of active categories carried through from the source

## Label mappings

### Fine‑grained (4‑class; preserves source fidelity)
`Not Toxic → 0`, `Hard to Say → 1`, `Toxic → 2`, `Very Toxic → 3`

**LLM‑jp mapping to fine:**  
`nontoxic → Not Toxic (0)`, `has_toxic_expression → Hard to Say (1)`, `toxic → Toxic (2)`  
*(LLM‑jp does not emit level 3.)*

### Coarse (binary‑ready)
`NonToxic` vs `Toxic`; Ambiguous (`Hard to Say`) is represented as missing/None in `label_int_coarse` so binary training can drop it cleanly.

### Optional ordinal view
For analysis, you can treat the fine labels as an **ordinal** 0–3 scale. This mainly benefits Inspection‑AI (LLM‑jp maps to 0–2).

## Week‑1 policy
We hand off Binary Strict: drop rows where `label_int_coarse` is missing (Ambiguous). We still keep the standardized files so tri‑state and ordinal ablations are trivial later.

## Week‑1 Deliverable
`data/processed/paired_native_romaji_inspection_ai_binary.csv` with columns:  
`id, text_native, text_romaji, label_int_coarse, label_text_fine, source`

- Input: the standardized Inspection‑AI CSV.  
- Ambiguous rows are removed (binary strict).

## Reproduction (commands, currently only tested on Inspection AI)
```bash
# Generate paired binary file
python scripts/pair_native_romaji.py   --infile  data/standardized/inspection_ai.csv   --outfile data/processed/paired_native_romaji_inspection_ai_binary.csv

python scripts/pair_native_romaji.py   --infile  data/standardized/llmjp.csv   --outfile data/processed/paired_native_romaji_llmjp_binary.csv
```
