# Agent Contract — VLM Claim-Analysis Pipeline

> Defines exactly what the VLM agent MUST do and the resilience rules around
> it. Written in Sprint 0. Implementation lands in Sprint 4 (`agent.py`),
> but the contract here is the source of truth for tests and review.

## 1. Architecture: two-pass

The agent uses a **two-pass** design. Pass 1 is cheap and text-only; Pass 2 is
multimodal and uses the targeted evidence requirement.

```
user_claim ─┐
            ├─▶ Pass 1 (text) ──▶ ClaimIntentExtraction
                              │      (claimed_damage_description,
                              │       issue_family,
                              │       claimed_object_part)
            │
images ─────┼─▶ Pass 2 (multimodal) ──▶ ClaimAnalysis
evidence ──┤      (uses issue_family to pick the evidence requirement,
history ──┘       constrained object_part list, history risk signal)
```

### 1.1 Pass 1 — intent extraction (text only)

- **Input:** `user_claim`, `claim_object`.
- **Output:** `ClaimIntentExtraction` (`claimed_damage_description`,
  `issue_family`, `claimed_object_part`).
- `issue_family` is a coarse phrase used to look up a row in
  `evidence_requirements.csv` (e.g. `dent or scratch`,
  `crack, broken, or missing part`).
- **Fallback on failure:** if Pass 1 errors after retries, the pipeline MUST
  NOT crash; set `issue_family = unknown` and proceed to Pass 2 with the
  general ("reviewability") requirement.

### 1.2 Pass 2 — multimodal analysis

- **Input:** all loadable images for the row, system prompt, user prompt.
- The user prompt embeds:
  - the extracted `claimed_damage_description` and `issue_family`,
  - the `minimum_image_evidence` text for the matched requirement,
  - the history-risk signal (`true`/`false`, NOT the raw history),
  - the **allowed `object_part` list for this `claim_object`** (so the model
    can only emit valid parts),
  - the allowed enums for `issue_type`, `severity`, `claim_status`,
    `risk_flags`.
- **Output:** `ClaimAnalysis` parsed from JSON
  (`response_mime_type = application/json`).

## 2. Model call parameters

| Parameter | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| `temperature` | `0` (every call) |
| Response type | JSON for structured calls |

> Determinism is a hard requirement (§6.2 of AGENTS.md). temperature=0 and
> deterministic postprocessing are how we get it.

## 3. Retry & rate-limit policy

1. **Retry library:** `tenacity`.
2. **Triggered exceptions:** `ResourceExhausted` (429), connection errors,
   transient server errors (5xx). Do NOT retry on auth/permission errors.
3. **Retry config:** `max_attempts = 5`, exponential backoff
   `multiplier = 2`, `min = 4 s`, `max = 60 s`. (`MAX_RETRIES = 5`.)
4. **Inter-call sleep:** `SLEEP_BETWEEN_CALLS = 4.0 s` between every model
   call to stay under the free-tier ~15 RPM ceiling.
5. **Per-row isolation:** a row failing must never abort the batch. Failures
   fall back to the safe-default dict (§4).

## 4. Fallback / safe default

If Pass 2 fails after all retries, the row is emitted with a safe default
dict that satisfies all consistency invariants:

| Field | Default |
|---|---|
| `evidence_standard_met` | `false` |
| `evidence_standard_met_reason` | `"Model analysis unavailable; image could not be evaluated."` |
| `risk_flags` | `manual_review_required` (plus `user_history_risk` if applicable) |
| `issue_type` | `unknown` |
| `object_part` | `unknown` |
| `claim_status` | `not_enough_information` |
| `claim_status_justification` | `"Automated review unavailable; manual review required."` |
| `supporting_image_ids` | `none` |
| `valid_image` | `false` |
| `severity` | `unknown` |

This default passes the §5 invariants in `data_contract.md` (valid_image
false ⇒ not supported).

## 5. Postprocessor contract

All model output passes through the postprocessor **before** being written.
The postprocessor is deterministic and is the sole place that guarantees
contract conformance. Its responsibilities:

1. **Enum validation + fuzzy fix.** Each enum field is normalized
   (lowercased, stripped) and matched; on near-miss use `difflib` to pick the
   closest canonical value; on failure fall back to `unknown` (or `none` /
   `not_enough_information` for the relevant fields).
2. **Object-bound part validation.** `object_part` is validated against the
   part list for THIS row's `claim_object` only. Out-of-type parts → `unknown`.
3. **Risk-flag split/validate/rejoin.** Split on `;`, strip/lower each token,
   drop tokens not in `VALID_RISK_FLAGS`, rejoin. Inject `user_history_risk`
   when the rule-based history signal is true.
4. **History flag injection (postprocessor, not model).** The model is
   instructed NOT to emit `user_history_risk`; the postprocessor adds it
   based on the rule-based signal from `data_loader.compute_history_risk`.
5. **Consistency enforcement** (see `behavior_spec.md` scenarios 7–8):
   - `valid_image = false` ⇒ `evidence_standard_met = false` and
     `claim_status ∈ {contradicted, not_enough_information}`.
   - `evidence_standard_met = false` ⇒ `claim_status ≠ supported`.
   - `claim_status = supported` ⇒ `valid_image = true` AND
     `evidence_standard_met = true`.
6. **Supporting-ID filtering.** `supporting_image_ids` tokens are
   intersected with the IDs parsed from this row's `image_paths`; empty → `none`.
7. **No nulls.** Replace any `None`/`NaN`/empty with the field sentinel.

## 6. Caching

- Caching is keyed by a deterministic hash of (model name, temperature,
  system prompt, user prompt, image bytes hash).
- A cache hit skips the network call entirely (counts as 0 model calls, 0 tokens).
- Used to avoid reprocessing completed claims after a crash/resume
  (satisfies US-006).
- Cache store lives under `code/.cache/` and is git-ignored.

## 7. What the model is NEVER allowed to decide

- The `user_history_risk` flag (rule-based only).
- Whether to override clear visual evidence with history (it can't).
- Final enum validity (the postprocessor is authoritative).
