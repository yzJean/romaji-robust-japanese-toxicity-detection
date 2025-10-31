# Romanization Notes

## Goal
Deterministically convert Japanese text to rōmaji so modeling can compare native vs romanized inputs on the same labels.  
The pipeline is **Unicode NFKC** → **pykakasi**. NFKC stabilizes compatibility/canonical variants; pykakasi handles Hiragana/Katakana/Kanji transliteration.  

## Converter config (Week‑1 defaults)
We enable conversion for H, K, and J to alphabet (`"a"`), choose Hepburn, and keep no spaces / no capitalization:

## Week‑1 policy choices (and why)
- **Romanization system:** **Hepburn** (most commonly used; friendlier to English‑speaking readers). We may compare Kunrei later.
- **Separators:** **Off** (`s=False`) for a clean baseline aligned with subword tokenizers. We’ll ablate `s=True` later. (pykakasi’s `s` inserts spaces.)
- **Capitalization:** **Off** (`C=False`) to avoid case variance in tokens.
- **Long vowels:** For Week‑1 we keep ASCII outputs (no macrons). Hepburn normally writes long vowels with macrons (ō, ū), and kana sequences like **ou/oo/uu** often realize long vowels; we’ll ablate a macron variant later.
- **Gemination (っ):** Doubled consonants; for っち Hepburn uses **tch** (e.g., まっちゃ → *matcha*).
- **Syllabic ん before vowels/y:** Hepburn uses **n’** apostrophe to disambiguate (e.g., 信用 → *shin’yō*). We accept pykakasi’s default; we can add a post‑pass if we find misses.

## Known quirks / what to watch
- **Loanwords with ー (chōonpu) in Katakana:** rōmaji length won’t match kana length; this is expected. ([Chōonpu — Wikipedia](https://en.wikipedia.org/wiki/Ch%C5%8Donpu))
- **Segmentation:** Edge cases in word segmentation if `s=True`; we start with `s=False` to avoid introducing word‑boundary noise.

## Sanity checklist for any new paired file
- No kana/kanji left in `text_romaji` (spot‑check a few lines).
- Reasonable length ratio between native and rōmaji (very short/empty conversions are red flags).
- Keep `id` and labels unchanged; only text differs (native vs rōmaji).

## References
- Unicode Normalization Forms (UAX #15): https://unicode.org/reports/tr15/
- pykakasi programming interface and mode switches: https://pykakasi.readthedocs.io/en/latest/api.html
- pykakasi documentation hub: https://pykakasi.readthedocs.io/
- Hepburn romanization (macrons, n’ before vowels/y, sokuon doubling incl. *tch*): https://en.wikipedia.org/wiki/Hepburn_romanization
- Chōonpu (prolonged sound mark “ー”): https://en.wikipedia.org/wiki/Ch%C5%8Donpu
