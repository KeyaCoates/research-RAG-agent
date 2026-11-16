# Sources

All documents in `corpus/` come from one source:

**Pilot's Handbook of Aeronautical Knowledge**, Federal Aviation Administration (FAA), US Department of Transportation.

- Edition: appears to be the 2016 edition (FAA-H-8083-25B). The PDF files were created in August 2016, and chapter 15 says drone rules were "expected by summer 2016". Please confirm the edition against the handbook's front page.
- Official page: https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/phak
- License: a US government publication, so it is not under copyright restrictions.

## Chapters used

| File | Chapter |
|---|---|
| `phak_ch04.md` | Chapter 4: Principles of Flight |
| `phak_ch05.md` | Chapter 5: Aerodynamics of Flight |
| `phak_ch08.md` | Chapter 8: Flight Instruments |
| `phak_ch09.md` | Chapter 9: Flight Manuals and Other Documents |
| `phak_ch15.md` | Chapter 15: Airspace |

## How the files were made

1. The text was extracted from the original chapter PDFs (kept locally in `data/raw/`, not committed).
2. It was converted to markdown. Figure labels and tables were flattened into text, so some headings and table contents are noisy.
3. No wording was changed by hand.

## Known limits

- This edition is out of date in places. For example, drone (unmanned aircraft) rules were still being written when it was published. Do not treat the corpus as current regulations or flight guidance.
- Figures, charts and diagrams from the PDFs are not included.
