# Data Contract — Multimodal Damage Claim Verification

> Single source of truth for all input/output schemas and allowed values.
> Every implementation module and every test MUST conform to this document.
> This spec is written FIRST; no implementation exists yet (Sprint 0).

## 1. Scope

The system reads claims from CSV, inspects submitted images with a VLM,
optionally consults user history and evidence requirements, and emits one
output row per input claim. This contract pins the exact shape of every CSV
column and every enum field so the submission is machine-evaluable.

Paths are relative to the repository root unless stated otherwise.

## 2. Input Contract

### 2.1 `dataset/claims.csv` — input-only test rows

| Field | Type | Notes |
|---|---|---|
| `user_id` | string | Key into `user_history.csv` |
| `image_paths` | string | One or more relative paths, semicolon-separated (`;`) |
| `user_claim` | string | Chat transcript; speaker turns split by `\|` |
| `claim_object` | enum | `car` \| `laptop` \| `package` |

### 2.2 `dataset/sample_claims.csv` — labeled examples

Same input fields as `claims.csv` plus the full set of expected output
columns (see §3.1). Used to develop against and to evaluate strategies.

### 2.3 `dataset/user_history.csv`

| Field | Type | Notes |
|---|---|---|
| `user_id` | string | Join key |
| `past_claim_count` | int | Total prior claims |
| `accept_claim` | int | Prior accepted |
| `manual_review_claim` | int | Prior manual reviews |
| `rejected_claim` | int | Prior rejections |
| `last_90_days_claim_count` | int | Recent activity |
| `history_flags` | string | Semicolon-separated flags, or `none` |
| `history_summary` | string | Free-text risk summary |

### 2.4 `dataset/evidence_requirements.csv`

| Field | Type | Notes |
|---|---|---|
| `requirement_id` | string | e.g. `REQ_CAR_BODY_PANEL` |
| `claim_object` | enum+ | `car` \| `laptop` \| `package` \| `all` |
| `applies_to` | string | Issue family, e.g. `dent or scratch` |
| `minimum_image_evidence` | string | Human-readable checklist text |

### 2.5 Image paths

- Format: relative to the **dataset root** (e.g. `images/test/case_001/img_1.jpg`).
- Multiple paths separated by `;`.
- The **image ID** is the filename without extension (e.g. `img_1`).
- Folders: `dataset/images/sample/` and `dataset/images/test/`.

## 3. Output Contract

### 3.1 `output.csv` — exact column order (14 columns)

```
user_id,
image_paths,
user_claim,
claim_object,
evidence_standard_met,
evidence_standard_met_reason,
risk_flags,
issue_type,
object_part,
claim_status,
claim_status_justification,
supporting_image_ids,
valid_image,
severity
```

Columns 1–4 (`user_id`, `image_paths`, `user_claim`, `claim_object`) are
copied verbatim from the input row. Columns 5–14 are produced by the system.

### 3.2 Field semantics

| Field | Meaning |
|---|---|
| `evidence_standard_met` | `true` if the image set is sufficient to evaluate the claim, else `false` |
| `evidence_standard_met_reason` | Short reason for the evidence decision |
| `risk_flags` | Semicolon-separated risk flags, or `none` |
| `issue_type` | Visible issue type (see §4.2) |
| `object_part` | Relevant object part, validated against the part list for THIS `claim_object` (see §4.4) |
| `claim_status` | `supported` \| `contradicted` \| `not_enough_information` |
| `claim_status_justification` | Concise image-grounded explanation; cite image IDs when helpful |
| `supporting_image_ids` | Image IDs supporting the decision, semicolon-separated; `none` if none suffice |
| `valid_image` | `true` if the image set is usable for automated review, else `false` |
| `severity` | `none` \| `low` \| `medium` \| `high` \| `unknown` |

### 3.3 Hard constraints (evaluable)

1. **No null/NaN/empty** in any output cell. Use sentinel strings (`none`,
   `unknown`) or `false` where appropriate, never blanks.
2. **No value outside the allowed enums** (see §4). Invalid values make the
   row score 0 on that field.
3. `supporting_image_ids` may **only** contain IDs derived from the row's own
   `image_paths` (filename without extension), or the literal `none`.
4. `object_part` must belong to the allowed part list for the row's
   `claim_object`. A laptop part on a car row is invalid.
5. Columns 1–4 must equal the input row verbatim (do not rewrite
   `image_paths` order).

## 4. Allowed Values (canonical enums)

### 4.1 `claim_object`
```
car | laptop | package
```

### 4.2 `issue_type`
```
dent | scratch | crack | glass_shatter | broken_part | missing_part |
torn_packaging | crushed_packaging | water_damage | stain | none | unknown
```
- `none` = relevant part visible and **no** issue present.
- `unknown` = issue/part cannot be determined.

### 4.3 `claim_status`
```
supported | contradicted | not_enough_information
```

### 4.4 `object_part` (per object type)
```
car:     front_bumper | rear_bumper | door | hood | windshield | side_mirror |
         headlight | taillight | fender | quarter_panel | body | unknown

laptop:  screen | keyboard | trackpad | hinge | lid | corner | port | base | body | unknown

package: box | package_corner | package_side | seal | label | contents | item | unknown
```
`unknown` is permitted for every object type.

### 4.5 `severity`
```
none | low | medium | high | unknown
```

### 4.6 `risk_flags` (semicolon-separated, or `none`)
```
none | blurry_image | cropped_or_obstructed | low_light_or_glare | wrong_angle |
wrong_object | wrong_object_part | damage_not_visible | claim_mismatch |
possible_manipulation | non_original_image | text_instruction_present |
user_history_risk | manual_review_required
```
- `none` is used **only** when no flags apply and is emitted alone.
- Otherwise, flags are joined by `;` (e.g. `blurry_image;wrong_angle`).

### 4.7 Booleans
```
evidence_standard_met: true | false
valid_image:           true | false
```
Emitted as lowercase strings `true` / `false`.

## 5. Consistency invariants

These are logical rules the postprocessor MUST enforce after the model
returns (see `behavior_spec.md` scenarios 7–8 and `agent_contract.md`
postprocessor rules):

1. If `valid_image = false` → `evidence_standard_met = false` and
   `claim_status ≠ supported` (must be `contradicted` or
   `not_enough_information`).
2. If `evidence_standard_met = false` → `claim_status ≠ supported`.
3. If `claim_status = supported` → `valid_image = true` AND
   `evidence_standard_met = true`.
4. If `claim_status = contradicted` and the part is visible with no issue,
   `issue_type = none`.
5. `supporting_image_ids` ⊆ image IDs parsed from this row's `image_paths`,
   unless it is the literal `none`.

## 6. Glossary

- **Image ID** — filename of an entry in `image_paths` without its extension.
- **Issue family** — coarse category used to look up the evidence
  requirement (e.g. `dent or scratch`, `crack, broken, or missing part`),
  derived in Pass 1 of the agent (see `agent_contract.md`).
- **History risk** — boolean derived **rule-based** from `user_history.csv`
  (never delegated to the model); drives the `user_history_risk` flag.
