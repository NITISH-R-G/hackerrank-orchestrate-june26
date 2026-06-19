# Behavior Spec — Multimodal Damage Claim Verification

> BDD-style Given/When/Then scenarios. These define the expected behavior the
> postprocessor enforces and the evaluation harness checks. Written in Sprint 0
> before any implementation; tests in `tests/` are derived from this file.

## Conventions

- Field values shown in backticks are the **canonical enum strings** from
  `data_contract.md`.
- "contains X" for `risk_flags` / `supporting_image_ids` means the token X
  appears among the semicolon-separated values.
- Image IDs are filenames without extension (e.g. an `image_paths` entry
  `images/test/case_001/img_1.jpg` → id `img_1`).

---

## SCENARIO 1 — Clear damage visible, matches claim

**Given** a user claims a car front-bumper dent
**When** the image clearly shows a dent on the front bumper
**Then**
- `claim_status` = `supported`
- `issue_type` = `dent`
- `object_part` = `front_bumper`
- `severity` ∈ {`low`, `medium`, `high`}
- `valid_image` = `true`
- `evidence_standard_met` = `true`
- `supporting_image_ids` contains the image ID showing the damage

---

## SCENARIO 2 — Image shows no damage

**Given** a user claims a laptop screen crack
**When** the image clearly shows the laptop screen with no visible cracks
**Then**
- `claim_status` = `contradicted`
- `issue_type` = `none`
- `object_part` = `screen`
- `valid_image` = `true`
- `evidence_standard_met` = `true` (the set was sufficient to conclude "no damage")

---

## SCENARIO 3 — Wrong object in image

**Given** a user claims package torn packaging
**When** the image shows a car, not a package
**Then**
- `claim_status` = `not_enough_information`
- `risk_flags` contains `wrong_object`
- `valid_image` = `false`  OR  (`valid_image` = `true` AND `evidence_standard_met` = `false`)

> Either branch is acceptable; the hard rule (scenario 7/8) is that a
> not_enough_information outcome must NOT be `supported`.

---

## SCENARIO 4 — Blurry, unusable image

**Given** a user claims any damage
**When** the image is too blurry to assess
**Then**
- `valid_image` = `false`
- `evidence_standard_met` = `false`
- `claim_status` = `not_enough_information`
- `risk_flags` contains `blurry_image`

---

## SCENARIO 5 — No images loadable

**Given** all `image_paths` are missing, unreadable, or corrupt
**Then**
- `valid_image` = `false`
- `evidence_standard_met` = `false`
- `claim_status` = `not_enough_information`
- `supporting_image_ids` = `none`

---

## SCENARIO 6 — High-risk user with a visually valid claim

**Given** a user whose `rejected_claim / past_claim_count > 0.3`
**When** the image clearly shows the claimed damage
**Then**
- `claim_status` = `supported`  (visual evidence wins; history never overrides clear visuals)
- `risk_flags` contains `user_history_risk`
- `risk_flags` contains `manual_review_required`

> Rationale: history adds *risk context* only. It must not flip a visually
> clear supported/contradicted outcome. It only ever *adds* risk flags and
> triggers manual review.

---

## SCENARIO 7 — `valid_image = false` consistency

**Given** `valid_image` is determined to be `false`
**Then**
- `claim_status` MUST NOT be `supported` (must be `contradicted` or `not_enough_information`)
- `evidence_standard_met` is **not forced** to `false`.

> **Data-grounded refinement.** Labeled `sample_claims.csv` contains rows
> where `valid_image = false` yet `evidence_standard_met = true` and
> `claim_status = contradicted` (e.g. a screenshot/non-original image that is
> still clear enough to conclude the claim mismatches). Therefore the only
> invariant the postprocessor enforces is "not supported". An earlier draft of
> this scenario asserted `evidence_standard_met` MUST be `false`; that is
> **refuted** by the ground-truth distribution and is NOT enforced.

---

## SCENARIO 8 — `evidence_standard_met = false` consistency

**Given** `evidence_standard_met` is `false`
**Then** `claim_status` MUST NOT be `supported`

> Contrapositive: `claim_status = supported` ⇒ `evidence_standard_met = true` AND `valid_image = true`.

---

## SCENARIO 9 — Multiple images, only some show damage

**Given** 3 images submitted, 2 show unrelated views and 1 shows the damage
**Then**
- `supporting_image_ids` contains **only** the 1 relevant image ID
- `supporting_image_ids` does NOT contain the 2 unrelated IDs
- every ID in `supporting_image_ids` is a valid ID parsed from `image_paths`

---

## SCENARIO 10 — No user history record

**Given** `user_id` is not found in `user_history.csv`
**Then**
- `history_risk` = `false` (internal signal)
- `risk_flags` does NOT contain `user_history_risk`
- the pipeline continues normally (no crash, no manual review triggered by history)

---

## Cross-cutting rules (apply to all scenarios)

These are enforced by the **postprocessor**, not the model, and override
model output when violated:

1. **Enum validity.** Every enum field is coerced to a canonical value via
   `difflib` fuzzy match; if unfixable, it falls back to `unknown` (or
   `none` for `issue_type`/`risk_flags`/`severity`).
2. **Part-object binding.** `object_part` is validated against the part list
   for the row's `claim_object`; cross-type parts are mapped to `unknown`.
3. **Risk-flag splitting.** `risk_flags` is split on `;`, each token
   validated/cleaned individually, then rejoined.
4. **History flag injection.** `user_history_risk` is added by the
   postprocessor when the rule-based `history_risk` is true — the model is
   never asked to emit it.
5. **Supporting-ID filtering.** `supporting_image_ids` tokens are filtered to
   IDs actually present in the row's `image_paths`; empties collapse to `none`.
6. **No nulls.** Any empty/None field is replaced by its sentinel.
