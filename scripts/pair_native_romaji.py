import argparse, csv, unicodedata, random
from pykakasi import kakasi

def to_romaji(s):
    s = unicodedata.normalize("NFKC", s)        # UAX #15
    k = kakasi()
    k.setMode("H","a")                          # Hiragana -> ascii
    k.setMode("K","a")                          # Katakana -> ascii
    k.setMode("J","a")                          # Kanji (JP) -> ascii
    k.setMode("r","Hepburn")                    # use Hepburn romanization
    k.setMode("s", False)                       # word seperators
    k.setMode("C", False)                       # capitalize words
    return k.getConverter().do(s)

ap = argparse.ArgumentParser()
ap.add_argument("--infile",  required=True)     # data/standardized/inspection_ai.csv
ap.add_argument("--outfile", required=True)     # data/processed/paired_inspection_ai_binary.csv
args = ap.parse_args()

rows = []
with open(args.infile, encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        # binary strict: drop Ambiguous (None)
        lic_raw = (row.get("label_int_coarse") or "").strip()
        if lic_raw.lower() in {"", "none", "null"}: 
            continue
        
        text_native = row["text_native"]
        
        text_romaji = to_romaji(text_native)
        
        rows.append({
            "id": row["id"],
            "text_native": text_native,
            "text_romaji": text_romaji,
            "label_int_coarse": int(lic_raw),
            "label_text_fine": row.get("label_text_fine"),
            "source": row.get("source")
        })

with open(args.outfile, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","text_native","text_romaji", "label_int_coarse","label_text_fine","source"])
    w.writeheader()
    w.writerows(rows)
