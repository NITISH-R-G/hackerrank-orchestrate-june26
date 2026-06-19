# Code — Multimodal Damage Claim Verification

This is the solution for the **HackerRank Orchestrate (June 2026)** multi-modal
evidence-review challenge. It verifies damage claims (car / laptop / package)
using submitted images, a short claim conversation, user history, and minimum
image-evidence requirements.

## Quick start

```bash
# from the repo root
pip install -r requirements.txt

# put your Gemini key in code/.env (gitignored)
echo "GEMINI_API_KEY=your_key" > code/.env
# optional extra keys to rotate across the 20/day free-tier quota:
# echo "GEMINI_API_KEY_2=..." >> code/.env

# produce output.csv for all 44 test claims (two-pass strategy)
cd code
python main.py

# evaluate both strategies on the 20 labeled samples, then write the report
python -m evaluation.main --full
python -m evaluation.report
```

`output.csv` is written to the **repo root** (one row per claim, exact 14-column
order). It is written **incrementally** (append-after-header), so a crash mid-run
keeps every prediction already made.

## Architecture

```
user_claim + images + claim_object
   │
   ├── Pass 1 (text-only): extract claimed damage + issue_family
   │       └─ on failure: fall back to issue_family=unknown, continue
   │
   ├── Pass 2 (multimodal): analyze images against the matched evidence
   │       requirement, with the object_part enum for THIS object type and
   │       the history-risk signal.
   │       └─ multi-image consistency check (same object across images?)
   │
   ├── Consistency validator: detect impossible combos; re-run Pass 2 once,
   │   then deterministic force-fix (e.g. severity=high + scratch -> medium)
   │
   └── Postprocessor (sole contract guarantor): alias-normalize enums,
       object-bound object_part, split+validate risk_flags, inject
       user_history_risk, filter supporting_image_ids, fill nulls, enforce
       invariants. Output is always contract-conforming.
```

The **postprocessor + validator + normalizer** are the deterministic safety net:
the model is never trusted to emit contract-conforming output on its own.

## Modules

| File | Purpose |
|---|---|
| `config.py` | Enums (allowed values), paths, model params; `SLEEP_BETWEEN_CALLS` env-overridable |
| `schema.py` | Pydantic models: `ClaimAnalysis`, `ClaimIntentExtraction`, `PipelineResult` |
| `prompts.py` | Pass 1 + Pass 2 + single-pass prompt builders (pure functions) |
| `data_loader.py` | Load CSVs; **rule-based** `compute_history_risk`; evidence-requirement lookup |
| `image_utils.py` | Parse `image_paths`, load images (skip-missing), image hashing |
| `cache.py` | Deterministic on-disk cache → resume after crash without reprocessing |
| `keypool.py` | Multi-key pool; marks daily-quota-exhausted keys dead, rotates live ones |
| `normalizer.py` | Alias-first enum normalization (`back bumper`→`rear_bumper`, …) |
| `validator.py` | Consistency checks + `force_fix`; configurable strict invariant flag |
| `postprocessor.py` | The contract guarantor — every model output passes through here |
| `agent.py` | Two-pass + single-pass VLM pipeline; retry, rotation, safe-default fallback |
| `main.py` | CLI entry point → `output.csv` (crash-safe, incremental) |
| `evaluation/metrics.py` | Exact-match, P/R/F1, 3×3 confusion, field error counts |
| `evaluation/main.py` | Runs both strategies on samples, auto-selects winner |
| `evaluation/report.py` | Generates `evaluation_report.md` from `summary.json` |

## Strategies

- **Strategy A — single-pass:** one multimodal call does intent + analysis.
- **Strategy B — two-pass (default for `output.csv`):** Pass 1 text intent
  extraction, then Pass 2 multimodal with the targeted evidence requirement.
  More calls, but deterministic evidence targeting and cleaner intent.

Selection rule (see `specs/evaluation_spec.md`): higher mean field accuracy
wins; ties go to two-pass for determinism.

## Reliability features

- **Retry:** `tenacity`-style loop; 5 attempts (7 for 5xx) with exponential
  backoff (×2, 13–90s) on 429/5xx only; 400/401 propagate immediately.
- **Daily-quota rotation:** a 429 on the PerDay quota marks the key dead and
  rotates to the next live key; transient per-minute/503 failures do NOT kill
  a key (the cursor resets each claim).
- **Caching:** SHA-256 keyed on (model, temperature, prompts, image hashes);
  a cache hit = 0 model calls, 0 tokens. Enables resume-after-crash.
- **Safe defaults:** if all keys/retries fail for a row, the row is filled
  with a contract-compliant default (`not_enough_information`, `unknown`,
  `manual_review_required`) and the run continues. `output.csv` is always
  complete and valid.
- **Per-row isolation:** one failing row never aborts the batch.

## Tests

171 deterministic tests (mocked model + real pipeline logic), 1 live test:

```bash
cd code
pytest                          # deterministic suite (default; excludes live)
pytest -m live                  # live Gemini call on a sample image
```

## Specs (spec-driven development)

`specs/` holds the contracts written **before** implementation:
`data_contract.md`, `behavior_spec.md` (10 BDD scenarios),
`agent_contract.md`, `evaluation_spec.md`.

## Operational notes

- Free-tier quota is **20 requests/day per project per model** (not per-minute).
  `keypool.py` exists precisely to spread a run across multiple projects.
- Default `SLEEP_BETWEEN_CALLS=13s` keeps a single key under the 5 RPM limit;
  set `SLEEP_BETWEEN_CALLS=4` (env) for multi-key runs.
- Cost: ~$0.05 to process the full 44-row test set at gemini-2.5-flash pricing
  (see `evaluation/evaluation_report.md` §6).
