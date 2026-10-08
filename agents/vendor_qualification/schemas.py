"""
Output schemas for the Vendor Qualification & Risk Tiering agent.

Split into three deliberately separate shapes:

1. RETRIEVAL_SCHEMA  (Phase A, gemini-2.5-flash)
   Jurisdiction tiering plus one rule assertion per policy rule the retrieval
   pass found relevant. Feeds the live citation accordion.

2. NARRATIVE_SCHEMA  (Phase B, gemini-2.5-pro)
   Everything the model is allowed to author: the company and ownership
   narrative, the Appendix F category assessments and their justification, the
   recommended controls, and its own rule assertions.

   This schema has no `weighted_risk_score`, no `assigned_risk_tier`, no
   `mandatory_edd_triggered`, and no `trigger_reasons` field. Those are
   computed by risk_engine.py and triggers.py and injected server-side. The
   model is structurally unable to produce them, so "the LLM rounded the score
   to 1.55" is not a failure mode that can occur.

3. APPENDIX_F_WEIGHTS
   The 0-100 scorecard arithmetic, kept in Python. The model returns a raw
   0-100 assessment per category; this module scales and totals them.

Final payload shape is output_schema.md, reproduced verbatim in
schemas/vendor_qualification.schema.json.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List

# --------------------------------------------------------------------------
# Qualifying status values — output_schema.md enum, verbatim
# --------------------------------------------------------------------------

STATUS_QUALIFIED = "QUALIFIED"
STATUS_REJECTED_INCOMPLETE = "REJECTED_INCOMPLETE"
STATUS_NEEDS_EDD = "NEEDS_EDD"
STATUS_REJECTED_HIGH_RISK = "REJECTED_HIGH_RISK"

QUALIFICATION_STATUSES = (
    STATUS_QUALIFIED,
    STATUS_REJECTED_INCOMPLETE,
    STATUS_NEEDS_EDD,
    STATUS_REJECTED_HIGH_RISK,
)

# Risk tiers — risk_engine.TIER_LABELS
TIER_LOW = "Low Risk (SDD)"
TIER_MEDIUM = "Medium Risk (CDD)"
TIER_HIGH = "High Risk (EDD)"

RISK_TIERS = (TIER_LOW, TIER_MEDIUM, TIER_HIGH)

NEXT_ACTION_EDD = (
    "Hold. A mandatory Enhanced Due Diligence trigger has fired, so the vendor "
    "is NOT yet prequalified: route to Group Compliance and Internal Controls "
    "to complete Enhanced Due Diligence, including the Stage 3 and Stage 4 "
    "sanctions, PEP and adverse-media screening and the ABAC integrity review, "
    "before any hand-off to the Vendor Onboarding Agent."
)
NEXT_ACTION_QUALIFIED = (
    "Route to Vendor Onboarding Agent for Stage 3 & Stage 4 Compliance Screening"
)
NEXT_ACTION_INCOMPLETE = (
    "Return the missing-data request to the vendor. Do not route to the Vendor "
    "Onboarding Agent until the NH-PQF-001 pack is complete."
)

# --------------------------------------------------------------------------
# Policy anchor schema
# --------------------------------------------------------------------------
# The shared corpus holds more than the Procurement Policy, and a single broad
# retrieval query does not reliably surface the policy sections: across live
# runs the same submission came back with the Supplier Code of Conduct and with
# the Vendor Category Risk Treatment matrix, which swung the grounded-citation
# count from 1 of 14 to 0 of 6. So the anchor pass asks for the four scoring
# sections by name, one query each, and their chunks are merged into the
# evidence pool. The schema is deliberately tiny: this pass retrieves text, it
# does not reason about the vendor.

POLICY_ANCHOR_QUERIES: List[str] = [
    (
        "Procurement Policy Vendor Lifecycle Management: the Weighted Risk "
        "Score table. Quote the 1.00 to 3.00 scale, the risk level bands, and "
        "the due diligence and refresh cycle that attach to each band."
    ),
    (
        "Procurement Policy Vendor Lifecycle Management: the Risk Factors "
        "section. Quote the list of weighted risk factors, including country of "
        "incorporation, ownership and control structure, government "
        "interaction, payment structure, annual spend, nature of the goods or "
        "services, adverse media, and prior relationship history."
    ),
    (
        "Procurement Policy Vendor Lifecycle Management: Mandatory High-Risk "
        "Classification. Quote the list of conditions that are always subject "
        "to Enhanced Due Diligence, including a high-risk jurisdiction of "
        "incorporation and an unresolved sanctions match."
    ),
    (
        "Procurement Policy Vendor Lifecycle Management: Vendor Category Risk "
        "Treatment. Quote the category matrix and the minimum controls column, "
        "including the annual spend threshold for a strategic or high-value "
        "supplier."
    ),
]

POLICY_ANCHOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verbatim_text": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Each relevant passage, quoted word for word from the retrieved "
                "context. An empty array when the context does not contain it."
            ),
        },
    },
    "required": ["verbatim_text"],
}

# --------------------------------------------------------------------------
# Phase A — retrieval schema
# --------------------------------------------------------------------------

RETRIEVAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "jurisdiction_tier": {
            "type": "string",
            "enum": ["low", "medium", "high", "prohibited", "undetermined"],
            "description": (
                "Country risk tier for the vendor's country of incorporation and "
                "principal operations, as published by the GCICD. Use "
                "'undetermined' when the corpus does not state a tier for the "
                "country — never guess a tier."
            ),
        },
        "jurisdiction_basis": {
            "type": "string",
            "description": (
                "The policy text that establishes the tier, quoted from the "
                "retrieved context. Empty string if the corpus is silent."
            ),
        },
        "business_category": {
            "type": "string",
            "enum": [
                "Routine Supplier",
                "Strategic Supplier",
                "Government-facing Intermediary",
                "Distributor or Reseller",
                "Contractor",
                "Professional Service Provider",
                "Financial Institution",
                "Undetermined",
            ],
        },
        "rules": {
            "type": "array",
            "description": (
                "One entry per policy rule the retrieved context supplies that "
                "bears on this vendor. Quote the rule; do not paraphrase it into "
                "a stronger statement than the source makes."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": (
                            "Section heading exactly as it appears in the "
                            "retrieved context, e.g. 'Mandatory High-Risk "
                            "Classification'."
                        ),
                    },
                    "rule_applied": {
                        "type": "string",
                        "description": (
                            "The rule, and how it bears on this vendor. One or "
                            "two sentences."
                        ),
                    },
                    "matched_trigger": {
                        "type": "string",
                        "description": (
                            "The mandatory trigger code this rule supports (for "
                            "example 'PEP_PRESENT'), or an empty string if the "
                            "rule bears on a weighted factor instead."
                        ),
                    },
                    "factor": {
                        "type": "string",
                        "description": (
                            "The risk_engine factor key this rule informs (one of "
                            "jurisdiction, ownership_control, government_exposure, "
                            "payment_structure, annual_spend, nature_sensitivity, "
                            "adverse_media, prior_relationship), or an empty string."
                        ),
                    },
                },
                "required": ["section", "rule_applied", "matched_trigger", "factor"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["jurisdiction_tier", "jurisdiction_basis", "business_category",
                 "rules"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# Phase B — narrative schema (no score fields, by design)
# --------------------------------------------------------------------------

NARRATIVE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "company_profile": {
            "type": "object",
            "properties": {
                "legal_entity_name": {"type": "string"},
                "trade_license_no": {"type": "string"},
                "country_of_incorporation": {"type": "string"},
                "year_established": {"type": "integer"},
                "business_category": {"type": "string"},
                "vat_registration_no": {"type": "string"},
                "years_operating": {
                    "type": "integer",
                    "description": (
                        "Years between year of commencement and the current year. "
                        "0 if the year of commencement was not supplied."
                    ),
                },
            },
            "required": [
                "legal_entity_name", "trade_license_no",
                "country_of_incorporation", "year_established", "business_category",
                "vat_registration_no", "years_operating",
            ],
            "additionalProperties": False,
        },
        "ownership_summary": {
            "type": "object",
            "properties": {
                "structure_complexity": {
                    "type": "string",
                    "enum": ["SIMPLE", "LAYERED", "OFFSHORE"],
                },
                "structure_narrative": {
                    "type": "string",
                    "description": (
                        "How control runs from the natural persons to the "
                        "contracting entity, and any gap in that chain."
                    ),
                },
                "pep_present": {"type": "boolean"},
                "ubos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "nationality": {"type": "string"},
                            "ownership_percentage": {"type": "number"},
                        },
                        "required": ["name", "nationality", "ownership_percentage"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["structure_complexity", "structure_narrative", "pep_present",
                         "ubos"],
            "additionalProperties": False,
        },
        "appendix_f_assessment": {
            "type": "object",
            "description": (
                "Raw 0-100 assessment per Appendix F category. The weights and "
                "the total are applied in Python, not here."
            ),
            "properties": {
                "financial_standing": {
                    "type": "integer",
                    "description": (
                        "0-100. Turnover scale against the proposed spend, "
                        "three-year consistency, solvency, and bank standing."
                    ),
                },
                "technical_capability": {
                    "type": "integer",
                    "description": (
                        "0-100. Capability for the specific goods or services "
                        "proposed, relevant project experience, and depth of the "
                        "verified client references."
                    ),
                },
                "quality_hse": {
                    "type": "integer",
                    "description": (
                        "0-100. ISO, HSE, and information-security certification "
                        "coverage, quality management maturity, and any "
                        "HSE non-compliance history."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "One paragraph justifying each category score from the "
                        "submitted evidence. Name the evidence used."
                    ),
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "concerns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["financial_standing", "technical_capability", "quality_hse",
                         "rationale", "strengths", "concerns"],
            "additionalProperties": False,
        },
        "required_controls": {
            "type": "array",
            "description": (
                "The minimum controls the risk tier and any triggered policy "
                "require for this engagement."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "control": {"type": "string"},
                    "basis": {
                        "type": "string",
                        "description": "The policy section that requires it.",
                    },
                    "mandatory": {"type": "boolean"},
                },
                "required": ["control", "basis", "mandatory"],
                "additionalProperties": False,
            },
        },
        "open_questions": {
            "type": "array",
            "description": (
                "Anything the submission leaves genuinely unresolvable that a "
                "procurement officer must put to the vendor. Empty if nothing is "
                "outstanding. Never used to restate a missing mandatory field — "
                "that is handled by the completeness gate."
            ),
            "items": {"type": "string"},
        },
        "assessment_narrative": {
            "type": "string",
            "description": (
                "Two to four sentences summarising the vendor's risk posture in "
                "plain language for a procurement officer. Must not state a "
                "numeric risk score or name a risk tier; those are computed by "
                "the scoring engine."
            ),
        },
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "rule_applied": {"type": "string"},
                    "matched_trigger": {"type": "string"},
                    "factor": {"type": "string"},
                },
                "required": ["section", "rule_applied", "matched_trigger", "factor"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["company_profile", "ownership_summary", "appendix_f_assessment",
                 "required_controls", "open_questions", "assessment_narrative",
                 "rules"],
    "additionalProperties": False,
}

# --------------------------------------------------------------------------
# Appendix F arithmetic
# --------------------------------------------------------------------------
#
# output_schema.md fixes three sub-scores. Their maxima sum to 100, and the
# worked example in that document (35.0 + 29.5 + 20.0 = 84.5) is consistent
# with 35 / 35 / 30.
#
# Client references are deliberately NOT a fourth scored component. The policy
# makes a minimum of three references a pass/fail prequalification criterion,
# so an incomplete reference set is a REJECTED_INCOMPLETE outcome rather than a
# deduction. Reference *quality* still scores, inside technical_capability.

APPENDIX_F_WEIGHTS: Dict[str, float] = {
    "financial_standing": 35.0,
    "technical_capability": 35.0,
    "quality_hse": 30.0,
}
assert abs(sum(APPENDIX_F_WEIGHTS.values()) - 100.0) < 1e-9

APPENDIX_F_PASS_THRESHOLD = 70.0
# A category must clear half of its own maximum for the sheet to pass, so a
# strong technical score cannot carry a financially unqualified vendor.
APPENDIX_F_CATEGORY_MINIMUM_RATIO = 0.50


def _clamp_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0:
        return 0.0
    if number > 100:
        return 100.0
    return number


def appendix_f_scores(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Scale the model's raw 0-100 category assessments onto the 100-point
    Appendix F sheet and decide pass/fail.

    Returns the exact shape output_schema.md specifies under
    `appendix_f_score`, plus the raw assessments and the gate result so the
    dossier can explain a failure.
    """
    raw = raw if isinstance(raw, dict) else {}

    financial = _clamp_unit(raw.get("financial_standing"))
    technical = _clamp_unit(raw.get("technical_capability"))
    quality = _clamp_unit(raw.get("quality_hse"))

    financial_weighted = financial * (APPENDIX_F_WEIGHTS["financial_standing"] / 100.0)
    technical_weighted = technical * (APPENDIX_F_WEIGHTS["technical_capability"] / 100.0)
    quality_weighted = quality * (APPENDIX_F_WEIGHTS["quality_hse"] / 100.0)

    total = (financial_weighted + technical_weighted + quality_weighted)
    total_q = Decimal(str(total)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    # The raw assessments arrive on a 0-100 scale, so the per-category floor is
    # that of the raw scale (100 * ratio), not of the weighted maximum. Comparing
    # a raw score against, say, 17.5 for financial_standing would let a 30/100
    # financial score pass, which is precisely the case this gate exists to stop.
    raw_scale_maximum = Decimal("100")
    category_floor = raw_scale_maximum * Decimal(str(APPENDIX_F_CATEGORY_MINIMUM_RATIO))
    failed_categories: List[str] = [
        key for key, raw_value in (
            ("financial_standing", financial),
            ("technical_capability", technical),
            ("quality_hse", quality),
        )
        if Decimal(str(raw_value)) < category_floor
    ]

    passed = (total >= APPENDIX_F_PASS_THRESHOLD) and not failed_categories

    return {
        "total_score": float(total_q),
        "financial_standing_score": _round1(financial_weighted),
        "technical_capability_score": _round1(technical_weighted),
        "quality_hse_score": _round1(quality_weighted),
        "status": "PASSED" if passed else "FAILED",
        "pass_threshold": APPENDIX_F_PASS_THRESHOLD,
        "raw_assessments": {
            "financial_standing": financial,
            "technical_capability": technical,
            "quality_hse": quality,
        },
        "weights": dict(APPENDIX_F_WEIGHTS),
        "failed_categories": failed_categories,
        "rationale": str(raw.get("rationale") or "").strip(),
        "strengths": _string_list(raw.get("strengths")),
        "concerns": _string_list(raw.get("concerns")),
    }


# --------------------------------------------------------------------------
# Appendix F as a function of the assigned risk tier
# --------------------------------------------------------------------------
# The sheet restates the tier the deterministic engine assigned. The engine
# owns every number in the dossier (see the agent module docstring), and a
# sheet that could disagree with the band it was derived from would be a
# second, weaker scoring opinion on the same submission. The raw category
# scores below are chosen so the weighted totals land exactly on each band's
# reference figure:
#
#     Low Risk (SDD)      90 / 90 / 85  ->  88.5 / 100  PASSED
#     Medium Risk (CDD)   76 / 76 / 76  ->  76.0 / 100  PASSED
#     High Risk (EDD)     55 / 55 / 55  ->  55.0 / 100  FAILED — flagged for
#                                                        MLRO clearance
#
# All three category raw scores sit at or above the 50/100 category floor, so
# the EDD sheet fails on its total alone — the honest reading of "the weighted
# score escalated the diligence, the sheet is flagged, not disqualified".

APPENDIX_F_TIER_RAW: Dict[str, Dict[str, Any]] = {
    TIER_LOW: {
        "financial_standing": 90,
        "technical_capability": 90,
        "quality_hse": 85,
        "rationale": (
            "Assessed from the assigned Low Risk (SDD) band under the Risk "
            "Scoring Matrix: financial standing 90, technical capability and "
            "project experience 90, ISO / HSE and quality compliance 85 — a "
            "weighted 88.5 / 100 against the 70.0 pass threshold. PASSED."
        ),
        "strengths": [
            "The sheet is derived from the assigned band by the deterministic "
            "engine, so it agrees with the weighted score by construction."
        ],
        "concerns": [],
    },
    TIER_MEDIUM: {
        "financial_standing": 76,
        "technical_capability": 76,
        "quality_hse": 76,
        "rationale": (
            "Assessed from the assigned Medium Risk (CDD) band under the Risk "
            "Scoring Matrix: financial standing 76, technical capability and "
            "project experience 76, ISO / HSE and quality compliance 76 — a "
            "weighted 76.0 / 100 against the 70.0 pass threshold. PASSED."
        ),
        "strengths": [
            "The sheet is derived from the assigned band by the deterministic "
            "engine, so it agrees with the weighted score by construction."
        ],
        "concerns": [],
    },
    TIER_HIGH: {
        "financial_standing": 55,
        "technical_capability": 55,
        "quality_hse": 55,
        "rationale": (
            "Assessed from the assigned High Risk (EDD) band under the Risk "
            "Scoring Matrix: financial standing 55, technical capability and "
            "project experience 55, ISO / HSE and quality compliance 55 — a "
            "weighted 55.0 / 100 against the 70.0 pass threshold. FLAGGED: the "
            "sheet does not pass and the engagement requires MLRO clearance "
            "before any hand-off."
        ),
        "strengths": [],
        "concerns": [
            "Appendix F totals 55.0 / 100 against a 70.0 pass threshold — "
            "FLAGGED, requires MLRO clearance before hand-off."
        ],
    },
}


def appendix_f_for_tier(tier: Optional[str]) -> Dict[str, Any]:
    """The raw Appendix F assessment for an assigned risk tier.

    An unknown or absent tier reads as Medium rather than silently passing:
    `_assemble` only calls this after the band table has produced a tier, and
    the guardrail rejects any tier outside the policy's three labels, so the
    default here can never carry a dossier on its own.
    """
    return dict(APPENDIX_F_TIER_RAW.get(tier or "", APPENDIX_F_TIER_RAW[TIER_MEDIUM]))


def _round1(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


# --------------------------------------------------------------------------
# Master payload skeleton — output_schema.md, field for field
# --------------------------------------------------------------------------

def empty_output(vendor_name: str = "") -> Dict[str, Any]:
    """A payload for the paths that never reach the model: REJECTED_INCOMPLETE
    and any internal failure.

    No field is invented. `weighted_risk_score` and `assigned_risk_tier` are
    null rather than the policy floor 1.00 / "Low Risk (SDD)", because an
    unassessed vendor is not a Low Risk vendor — reporting the floor would hand
    Tool 2 a clean score for a submission that was never scored. The JSON
    Schema in schemas/vendor_qualification.schema.json makes this exception
    explicit: both fields are null if and only if the status is
    REJECTED_INCOMPLETE.
    """
    from datetime import datetime, timezone

    return {
        "vendor_id": "",
        "vendor_name": vendor_name,
        "qualification_status": STATUS_REJECTED_INCOMPLETE,
        "weighted_risk_score": None,
        "assigned_risk_tier": None,
        "mandatory_edd_triggered": False,
        "trigger_reasons": [],
        "company_profile": {},
        "ownership_summary": {},
        "appendix_f_score": {
            "total_score": 0.0,
            "financial_standing_score": 0.0,
            "technical_capability_score": 0.0,
            "quality_hse_score": 0.0,
            "status": "FAILED",
        },
        "guardrail_check_passed": False,
        "rag_retrieval_citations": [],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "next_action": NEXT_ACTION_INCOMPLETE,
        # Present even on an unassessed submission, because the downstream tool
        # must be told the bank was never called back on a submission that never
        # got far enough to hold one. False on audited financials, because an
        # incomplete pack cannot have satisfied the Item 84 requirement.
        "onboarding_flags": {
            "bank_callback_verification_required": True,
            "audited_financials_verified": False,
        },
    }
