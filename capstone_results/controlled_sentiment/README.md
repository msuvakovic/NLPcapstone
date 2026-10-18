# Controlled capstone evidence bundle

This directory consolidates compact, measured artifacts used by the final
capstone report. Rebuild the derived files with:

```bash
python3 tools/build_capstone_report.py
```

## Provenance

- `source/` contains selected JSON artifacts copied without modification from
  `origin/albert` commit `98a911810a5defcbd772af22035ed46b08274560`.
- The controlled sentiment experiments used real Gemma 4 31B cloud responses.
- The selected Qwen 2.5 7B reasoning logs remain in
  `prompt_opt_harness/prompt_opt_harness/logs/`.
- No mock-backend accuracy is included in `summary.json` or `report.md`.
- No new model calls were made during consolidation on October 18, 2026.

The compact bundle intentionally omits large per-example prediction files. They
remain recoverable from the source commit and paths named in the report. The
committed protocol, search, verification, and summary files are sufficient to
reconstruct the reported aggregate metrics and resource accounting.

Derived files:

- `protocol.json`: scope, inclusion rules, and experiment roles;
- `source_manifest.json`: original Git paths plus SHA-256 hashes;
- `summary.json`: machine-readable consolidated results;
- `requirements.json`: slide-deck completion checklist;
- `report.md`: final Q1–Q3 analysis and recommendations.