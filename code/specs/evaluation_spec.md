# Evaluation Spec — Strategy Comparison & Operational Analysis

> Written in Sprint 0. The evaluation harness (Sprint 5, `evaluation/`) must
> conform to this. Goal: justify the chosen approach on labeled data before
> running the test set.

## 1. Objective

Compare two analysis strategies on `dataset/sample_claims.csv`, pick the
better one, and run ONLY that strategy on `dataset/claims.csv` to produce
`output.csv`. Report cost/latency/rate-limit considerations.

## 2. Strategies

### Strategy A — single-pass
- **One** multimodal call per claim: images + raw `user_claim` + claim_object
  in a single prompt. The model extracts intent and analyzes visuals together.
- Fewer calls, larger prompt, no targeted evidence requirement.

### Strategy B — two-pass (default pipeline)
- **Two** calls per claim: Pass 1 text-only intent extraction → Pass 2
  multimodal analysis with the targeted `minimum_image_evidence` requirement
  and constrained `object_part` list (see `agent_contract.md`).
- More calls, more precise prompt, deterministic evidence targeting.

## 3. Scored fields

Both strategies are scored against the labeled `sample_claims.csv` on these
fields (per-field **exact match** accuracy):

1. `claim_status`
2. `issue_type`
3. `object_part`
4. `severity`
5. `valid_image`
6. `evidence_standard_met`

Comparison rules:
- Normalization before scoring: lowercase, strip. Booleans compared as
  lowercase strings.
- `supporting_image_ids` and `risk_flags` are reported for diagnostics but
  are **not** primary accuracy fields (set-membership semantics make exact
  match noisy); a relaxed subset check is reported alongside.

## 4. Metric

- **Primary:** per-field exact-match accuracy (0–1) for each of the 6 fields,
  per strategy. Also report the mean across fields.
- **Selection rule:** choose the strategy with the higher mean accuracy. On a
  tie, prefer Strategy B (two-pass) for determinism and targeted evidence.
- Report the confusion breakdown for `claim_status` (3×3) for each strategy.

## 5. Required report — `evaluation/evaluation_report.md`

The report MUST include:

1. **Per-field accuracy table** for Strategy A and Strategy B (6 fields + mean).
2. **`claim_status` confusion** (3×3) for each strategy.
3. **Winning strategy** and one-paragraph rationale referencing the numbers.
4. **Operational analysis** (see §6).
5. **Reproducibility note:** exact command(s) to re-run evaluation and to
   regenerate `output.csv`.

## 6. Operational analysis (required)

Report approximations for **both sample and full test set**:

- **Model calls:** count of Pass 1 + Pass 2 calls actually made (cache hits
  excluded), per strategy and per set.
- **Tokens:** approximate input + output tokens (from `usage_metadata` where
  available; otherwise estimated).
- **Images processed:** number of images sent to the model.
- **Cost:** estimated USD to process the full test set, with explicit pricing
  assumptions stated (e.g. gemini-2.5-flash $X/M input, $Y/M output).
- **Latency/runtime:** wall-clock time for sample and (projected) test set,
  noting the `SLEEP_BETWEEN_CALLS` rate-limit throttle.
- **Rate limits (TPM/RPM):** state the assumed ceiling (15 RPM free tier),
  and describe the throttling/batching/caching/retry strategy from
  `agent_contract.md` §3.
- **Optimizations applied:** caching, single-pass-vs-two-pass trade-off,
  resume-after-crash, any prompt compaction.

## 7. Determinism & reproducibility

- `temperature = 0` everywhere.
- Random seed fixed for any non-model logic.
- Evaluation reads `dataset/sample_claims.csv` and writes a metrics summary;
  it never mutates the dataset.
- The final `output.csv` is produced by running the **winning strategy** on
  `dataset/claims.csv`, end to end, in a single deterministic pass.

## 8. Acceptance (Sprint 5)

- [ ] Both strategies runnable on `sample_claims.csv`.
- [ ] Per-field accuracy table present and correct.
- [ ] Winning strategy selected by the documented rule.
- [ ] Operational analysis section complete with pricing assumptions.
- [ ] `output.csv` regenerated using the winning strategy only.
