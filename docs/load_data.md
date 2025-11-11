# Data Standardization Guide — `scripts/load_data.py`

## Purpose
Standardize two Japanese toxicity datasets into a single, ML‑ready schema while preserving provenance and uncertainty:
- **Inspection AI – Japanese Toxic Dataset**: multi‑annotator **vote counts** per fine label **and** per‑category vote counts. No single “gold” label is provided.
- **LLM‑jp Toxicity Dataset v2**: a single **label** per text plus seven binary category flags (“yes”/“no”).

This document records design choices and exactly how `scripts/load_data.py` transforms each source.

---

## Standardized Output Schema (CSV)

| Column | Type | Meaning |
|---|---|---|
| `id` | string | Unique row id with source prefix (`insp_…`, `llmjp_…`). |
| `text_native` | string | Original Japanese text. |
| `label_text_fine` | string | Fine label in our 4‑class space: `Not Toxic`, `Hard to Say`, `Toxic`, `Very Toxic`. |
| `label_int_fine` | int | 0,1,2,3 for the above (see `FINE_MAP`). |
| `label_text_coarse` | string | Coarse label: `NonToxic`, `Ambiguous`, `Toxic`. |
| `label_int_coarse` | int/None | `0` for NonToxic, `1` for Toxic, `None` for Ambiguous. |
| `label_conf` | float/None | Confidence of chosen fine label (majority share) when available; else `None`. |
| `low_confidence` | bool | Heuristic flag: `label_conf < 0.6` for Inspection‑AI rows; `False` for LLM‑jp. |
| `categories` | string (JSON list) | Active category names (Inspection‑AI: any `category_*` with count > 0; LLM‑jp: any flag with `"yes"`). |
| `source` | string | `inspection-ai` or `llm-jp`. |
| `split` | string | Dataset split tag (`unsplit` unless you assign splits later). |
| `orig_label` | string | Source label string (or majority label string for Inspection‑AI). |
| `orig_meta` | string (JSON) | Raw metadata for auditability (votes, `annotation_num`, per‑category counts, original fields). |

> **Why both fine and coarse?** Fine preserves severity nuance (`Very Toxic` vs `Toxic`); coarse provides a robust binary + ambiguous view useful for simpler tasks and evaluation. LLM‑jp’s `has_toxic_expression` aligns to our `Hard to Say` (Ambiguous) bucket.

---

## Label Mappings (constants)

```python
FINE_MAP   = {"Not Toxic":0, "Hard to Say":1, "Toxic":2, "Very Toxic":3}
COARSE_MAP = {"Not Toxic":"NonToxic", "Hard to Say":"Ambiguous", "Toxic":"Toxic", "Very Toxic":"Toxic"}
```

- `label_int_coarse`: `0` if fine is `Not Toxic`; `1` if fine is `Toxic`/`Very Toxic`; `None` if fine is `Hard to Say`.
- Downstream code must handle `None` (or later remap to `-1` if strictly numeric classes needed).

---

## Adapter: `adapt_inspection_ai(csv_path, out_path)`

**Input**: CSV per Inspection‑AI schema with columns: `id`, `text`, vote counts for `Not Toxic`/`Hard to Say`/`Toxic`/`Very Toxic`, `annotation_num`, and multiple `category_*` columns (each an **integer vote count**, not boolean).

**Steps**
1. **Read votes** into a dict: `{fine_label: count}` for the four fine labels.
2. **Total annotations**: use `annotation_num` if present; else `sum(votes.values())`.
3. **Tie‑aware majority selection (fine label)**:
   - Collect all labels with the **max** count.
   - If **> 1** winners:
     - If all winners map to the **same coarse bucket** and that bucket is `Toxic` (e.g., `Toxic` vs `Very Toxic`), pick `Toxic` as the fine label.
     - Otherwise mark **`Hard to Say`** (ambiguous).
   - Else (single winner): use that label.
4. **Confidence**: `label_conf = votes[fine] / ann_n` (or `None` if `ann_n == 0`).
5. **Low‑confidence flag**: `True` if `label_conf < 0.6`.
6. **Categories**:
   - Build `cat_votes = {name: count}` from all `category_*` fields.
   - Set `categories` to a JSON list of names with `count > 0`.
   - Persist `cat_votes` inside `orig_meta` for full auditability.
7. **Provenance**:
   - `orig_label`: the chosen **fine** label string (post majority/tie policy).
   - `orig_meta`: JSON with `votes`, `annotation_num`, and `category_votes` to retain full raw signal.

**Rationale**
- Majority vote converts **vote vectors** to a single fine label, mirroring common practice.
- The tie policy preserves severity within the toxic bucket while treating cross‑bucket ties as ambiguous.
- We keep confidence + raw votes so you can filter/weight low‑agreement items later.
- Categories are multi‑label; we keep both a **present/absent** list and the raw **counts** for analysis.

---

## Adapter: `adapt_llmjp(jsonl_path, out_path)`

**Input**: JSONL with fields `text`, **single** `label` ∈ {`toxic`,`nontoxic`,`has_toxic_expression`}, and seven binary attributes `obscene`,`discriminatory`,`violent`,`illegal`,`personal`,`corporate`,`others` (strings `"yes"`/`"no"`).

**Steps**
1. **Text**: `obj["text"]` (fallback `sentence` if present).
2. **Fine mapping**:
   - `nontoxic` → `Not Toxic`
   - `has_toxic_expression` → `Hard to Say`
   - `toxic` → `Toxic`
3. **Coarse mapping**: via `COARSE_MAP`.
4. **Confidence**: `None` (no votes provided).
5. **Low‑confidence**: `False`.
6. **Categories**: include any attribute whose value is `"yes"` (lowercased).
7. **Provenance**:
   - `orig_label`: the raw `label` string (lowercased).
   - `orig_meta`: JSON of all original fields except the text keys for compactness.

**Rationale**
- Aligns LLM‑jp’s three‑way label to our fine/coarse spaces and treats `has_toxic_expression` as ambiguous.
- Keeps diagnostic categories (reason codes) as a list for multi‑label analysis.

---

## CLI Usage

```bash
# Inspection AI → standardized CSV
python scripts/load_data.py --inspection data/raw/inspection_ai/labels.csv   --out data/standardized/inspection_ai.csv

# LLM‑jp v2 → standardized CSV
python scripts/load_data.py --llmjp data/raw/llmjp/toxicity_dataset.jsonl   --out data/standardized/llmjp.csv
```

---

## Determinism & Reproducibility
- **IDs**: `insp_{row['id']}` and `llmjp_{i}` (index within file).
- **Tie policy** is explicit; no dependence on Python dict order.
- **Rounding**: `label_conf` rounded to **3 decimals** for stable output.
- **Provenance**: raw votes, per‑category vote counts, and original fields are preserved in `orig_meta`.

---

## Validation Checks (recommended, not yet implemented)
- **Inspection‑AI**: `sum(votes.values()) == annotation_num`. Flag rows where totals disagree.
- **LLM‑jp**: If `label ∈ {toxic, has_toxic_expression}`, at least one category should be `"yes"`; `nontoxic` rows should have all `"no"` except possible PII cases (`personal`/`corporate`).

---

## References
- Inspection AI — Japanese Toxic Dataset (schema & subset): https://github.com/inspection-ai/japanese-toxic-dataset
- LLM‑jp Toxicity Dataset v2 (news/release): https://llm-jp.nii.ac.jp/en/news/release-of-the-japanese-toxic-text-dataset-llm-jp-toxicity-dataset-v2/
- LLM‑jp Toxicity Dataset v2 (GitLab project): https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-toxicity-dataset-v2
- LLM‑jp Toxicity Dataset (HF viewer – shows `label` + yes/no flags): https://huggingface.co/datasets/p1atdev/LLM-jp-Toxicity-Dataset
