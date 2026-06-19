"""Pydantic schemas for the VLM claim-analysis pipeline.

Mirrors ``specs/agent_contract.md`` §1 and ``specs/data_contract.md`` §3.
These models are the typed boundary between the model output and the
postprocessor.
"""
from pydantic import BaseModel


class ClaimAnalysis(BaseModel):
    """Pass 2 multimodal output. Every field the model is asked to produce."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: str


class ClaimIntentExtraction(BaseModel):
    """Pass 1 text-only output."""

    claimed_damage_description: str
    issue_family: str
    claimed_object_part: str


class PipelineResult(BaseModel):
    """Final, postprocessor-validated row written to output.csv.

    Mirrors ``config.OUTPUT_COLUMNS`` exactly (tests assert every column is
    present in ``model_dump()``).
    """

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: str
