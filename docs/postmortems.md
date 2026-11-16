# Postmortems

## 2026-11-15: eval crashed on Windows with UnicodeDecodeError

Symptom: run_eval crashed while reading corpus files on Windows.
Cause: read_text() was called without an encoding. Python used the system default (cp1252 on Windows), but the files are UTF-8. It worked on Linux, where the default is UTF-8, so it was not caught.
Fix: always pass encoding="utf-8" when reading and writing text files.
Prevention: tests/test_encoding.py fails if any read_text, write_text or text-mode open in src/ omits an encoding, and checks a non-ASCII round trip. Still to do: also run CI on a Windows agent.

## 2026-11-15: chunker cut long paragraphs mid-sentence

Symptom: found while designing chunk-level scoring. Converted PDF paragraphs were one very long line, and the chunker hard-split at exactly max_chars, often in the middle of a sentence, so a supporting passage could be split across two chunks and neither chunk would contain it.
Fix: split on sentence boundaries, never cut inside a sentence, and repeat the last sentence at the start of the next chunk (overlap).
Prevention: test_long_paragraph_is_not_cut_mid_sentence.
Note: the first eval numbers (document-level, before this fix) were inflated for a second reason as well, see the results table in the README.
