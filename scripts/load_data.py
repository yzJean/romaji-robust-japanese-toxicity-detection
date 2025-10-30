# scripts/load_data.py
import argparse, json, csv, pathlib

FINE_MAP = {"Not Toxic":0, "Hard to Say":1, "Toxic":2, "Very Toxic":3}
COARSE_MAP = {"Not Toxic":"NonToxic", "Hard to Say":"Ambiguous",
              "Toxic":"Toxic", "Very Toxic":"Toxic"}

def adapt_inspection_ai(csv_path, out_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            votes = {k:int(row[k]) for k in ["Not Toxic","Hard to Say","Toxic","Very Toxic"]}
            ann_n = int(row.get("annotation_num") or 0) or sum(votes.values())
            max_vote = max(votes.values())
            winners = [label for label, v in votes.items() if v == max_vote]
            if len(winners) > 1:
                coarse_winners = set(COARSE_MAP[label] for label in winners)
                if len(coarse_winners) == 1 and "Toxic" in coarse_winners:
                    label_text_fine = "Toxic"
                else:
                    label_text_fine = "Hard to Say"
            else:
                label_text_fine = winners[0]
            label_int_fine = FINE_MAP[label_text_fine]
            label_conf = (votes[label_text_fine] / ann_n) if ann_n else None
            low_confidence = (label_conf is not None and label_conf < 0.6)
            cat_votes = {
                k.replace("category_",""): int(v)
                for k, v in row.items() if k.startswith("category_")
            }
            cats = [name for name, c in cat_votes.items() if c > 0]
            
            out = {
                "id": f"insp_{row['id']}",
                "text_native": row["text"],
                "label_text_fine": label_text_fine,
                "label_int_fine": label_int_fine,
                "label_text_coarse": COARSE_MAP[label_text_fine],
                "label_int_coarse": 0 if label_text_fine == "Not Toxic"
                                   else (1 if label_text_fine in ["Toxic","Very Toxic"] else None),
                "label_conf": round(label_conf,3) if label_conf is not None else None,
                "low_confidence": low_confidence,
                "categories": json.dumps(cats, ensure_ascii=False),
                "source": "inspection-ai",
                "split": "unsplit",
                "orig_label": label_text_fine,
                "orig_meta": json.dumps({"votes":votes, "annotation_num":ann_n, "category_votes": cat_votes}, ensure_ascii=False),
            }
            rows.append(out)
    write_csv(rows, out_path)

def adapt_llmjp(jsonl_path, out_path):
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            txt = obj.get("text") or obj.get("sentence") or ""
            raw_lbl = (obj.get("label") or "").strip().lower()
            if raw_lbl == "nontoxic":
                fine = "Not Toxic"
            elif raw_lbl == "has_toxic_expression":
                fine = "Hard to Say"
            elif raw_lbl == "toxic":
                fine = "Toxic"
            else:
                fine = "Hard to Say"
            cats = []
            for attr in ["obscene", "discriminatory", "violent", "illegal", "personal", "corporate", "others"]:
                if obj.get(attr, "").strip().lower() == "yes":
                    cats.append(attr)
            out = {
                "id": f"llmjp_{i}",
                "text_native": txt,
                "label_text_fine": fine,
                "label_int_fine": FINE_MAP[fine],
                "label_text_coarse": COARSE_MAP[fine],
                "label_int_coarse": 0 if fine == "Not Toxic"
                                 else (1 if fine == "Toxic" else None),
                "label_conf": None,
                "low_confidence": False,
                "categories": json.dumps(cats, ensure_ascii=False),
                "source": "llm-jp",
                "split": "unsplit",
                "orig_label": raw_lbl,
                "orig_meta": json.dumps({k:v for k,v in obj.items() if k not in ["text","sentence"]}, ensure_ascii=False),
            }
            rows.append(out)
    write_csv(rows, out_path)

def write_csv(rows, out_path):
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = ["id","text_native","label_text_fine","label_int_fine",
            "label_text_coarse","label_int_coarse","label_conf",
            "low_confidence","categories","source","split","orig_label","orig_meta"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspection")
    ap.add_argument("--llmjp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.inspection:
        adapt_inspection_ai(args.inspection, args.out)
    elif args.llmjp:
        adapt_llmjp(args.llmjp, args.out)
    else:
        ap.error("Provide --inspection or --llmjp")
