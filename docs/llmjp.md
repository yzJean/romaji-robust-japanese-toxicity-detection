# LLM-jp Toxicity Dataset v2 (Overview)

## What is it

Public Japanese toxicity dataset by the [LLM-jp project](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-toxicity-dataset-v2), designed to benchmark toxicity detection for Japanese language models. Each entry provides a single overall toxicity label and several binary category attributes describing the type of toxicity.
## Columns (as defined by the repo)

Each row represents one text sample and its annotations:
- `label` – overall toxicity label of the text. Possible values:
  - `toxic`: The text is toxic.
  - `nontoxic`: The text is free from toxicity.
  - `has_toxic_expression`: The text contains potentially toxic expressions but is not toxic overall.
- Category flags (yes/no):
  - `obscene`, `discriminatory`, `violent`, `illegal`, `personal`, `corporate`, `others` – indicates presence of the corresponding harmful expression type.
- `text` - the sentence under evaluation

According to dataset rules:
- Texts labeled `toxic` or `has_toxic_expression` always have **atleast one** category attirbute marked `"yes"`.
- Texts labeled `nontoxic` generally have all categories marked `"no"`, except in cases with personally identifiable information (PII), where `personal` or `corporate` may still be `"yes"`.

## How the labels work in the dataset

Unlike the Inspection AI dataset, LLM-jp already provides one explicit label per sample. Theres no votes or confidence scores – `label` itself is the authorititative source of truth for toxicity classification.

The additional binary attributes specify **why** a text was considered toxic (e.g. because it was obscene or discriminatory).

## Nuances of this dataset

- **Single-label structure.** Each text already has one defined toxicity label; no derivation needed and no confidence metric is computed.
- **Ambiguity middle class.** The `has_toxic_expression` label captures borderline or uncertain cases, which is roughly analogous to "Hard to Say" in our standardized schema.
- **Category flags are diagnostic.** The seven yes/no attributes explain which kinds of harmful expression occur. They are independent of the main label.
- **Consistent format.** All flags use `"yes"`/`"no"` strings which simplifies parsing. 

## How we use it (Our implementation)

This is explained in the `docs/load_data.md` file.