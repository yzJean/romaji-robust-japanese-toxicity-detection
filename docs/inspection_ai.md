# Inspection AI - Japanese Toxic Dataset (Overview)

## What is it

Public dataset + schema for Japanese toxicity detection from Inspection AI (Apache-2.0). [Repo](https://github.com/inspection-ai/japanese-toxic-dataset/tree/main) includes a schema and a small subset CSV for reference

## Columns (as defined by the repo)

Each row is one sentence with multi-annotator votes and optional category flags:
- `id` – sentence ID
- `text` – the sentence
- Vote columns (fine labels):
  - `Not Toxic`, `Hard to Say`, `Toxic`, `Very Toxic` – counts of annotator votes for each label. (These are not single labels, it just represents how many people picked each class)
- Category vote columns (integers ≥ 0):
  - `category_卑語` (Dignity), `category_差別` (Discrimination), `category_迷惑行為` (Harassment), `category_猥褻` (Obscenity), `category_出会い・プライバシー侵害` (Privacy), `category_違法行為` (Illegal), `category_偏向表現` (Bias).  
*Interpretation:* values are annotator counts. A category is "present" if its count > 0.
- `annotation_num` - number of annotators for that row

## How the labels work in the dataset

The dataset does not give a single "gold" label to the sentences, instead it gives a vote count per fine label. Users typically derive a label (in our case, through majority vote) and may compute a confidence (majority votes / total votes).

## Nuances of this dataset

- **Ambiguity is explicit.** `Hard to Say` is a first-class label in the schema. We are not forcing everything to a binary toxic/non-toxic decision to preserve uncertainty.
- **Ties can occur.** Since labels are vote counts, equal maxima (e.g. `Toxic` = 2 and `Not Toxic` = 2) is possible. We define a tie-break policy
- **Annotator counts vary.** `annotator_num` should be used to understand how strong a majority is (in our case, confidence labels).
- **Categories are multi-label.** Category flags are independent of the fine label votes and can be used for sub-type analysis (e.g. discriminatory vs obscenity.)

## How we use it (Our implementation)

This is explained in the `docs/load_data.md` file.