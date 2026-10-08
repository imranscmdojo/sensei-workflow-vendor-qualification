"""
Output-contract enforcement for the Vendor Qualification dossier.

Two independent layers, because they catch different things:

1. `validate_dossier` runs the strict JSON Schema in
   schemas/vendor_qualification.schema.json (draft-07), which is
   output_schema.md reproduced field for field plus the documented
   REJECTED_INCOMPLETE null exception. Uses `jsonschema` when it is
   installed; the service still runs without it, so the layer is reported as
   skipped rather than silently passing.

2. `check_invariants` re-derives the dossier's arithmetic and cross-field
   consistency in plain Python. It needs no dependency and it is the layer
   that actually matters for a compliance artefact: it is what proves the
   score was not tampered with between the scoring engine and the response,
   that the tier matches the score, that a mandatory EDD override is always
   reported as High Risk, and that Appendix F's components add up to its
   total.

Layer 2 runs even when layer 1 is unavailable. A violation is reported rather
than raised: the caller downgrades guardrail_check_passed to False and returns
the dossier, so a procurement officer still sees the findings and a developer
still gets the violation list. It never hands back a dossier that claims the
guardrail passed while the arithmetic disagrees.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import risk_engine

try:  # optional: strict schema validation when the dependency is present
    import jsonschema
    from jsonschema import Draft7Validator
    _JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on a slim install
    jsonschema = None  # type: ignore[assignment]
    Draft7Validator = None  # type: ignore[assignment]
    _JSONSCHEMA_AVAILABLE = False


SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "schemas",
    "vendor_qualification.schema.json",
)

APPENDIX_F_PASS_THRESHOLD = 70.0
APPENDIX_F_MAX = {
    "financial_standing": 35.0,
    "technical_capability": 35.0,
    "quality_hse": 30.0,
}
APPENDIX_F_CATEGORY_MINIMUM_RATIO = 0.50
EDD_FLOOR = 2.50

_SCHEMA_CACHE: Optional[Dict[str, Any]] = None


def json_schema_available() -> bool:
    return _JSONSCHEMA_AVAILABLE


def load_schema() -> Dict[str, Any]:
    """The dossier schema, read once."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
            _SCHEMA_CACHE = json.load(handle)
    return _SCHEMA_CACHE


def _validator() -> Optional[Any]:
    if not _JSONSCHEMA_AVAILABLE:
        return None
    return Draft7Validator(load_schema())


def validate_dossier(dossier: Dict[str, Any]) -> List[str]:
    """Schema-level errors, as readable strings. Empty list means valid.

    Returns a single-element list describing the skip when `jsonschema` is not
    installed, so the caller can never mistake "not checked" for "passed".
    """
    validator = _validator()
    if validator is None:
        return [
            "schema validation skipped: the 'jsonschema' package is not "
            "installed, so only the deterministic invariants were checked"
        ]
    errors = []
    for error in sorted(validator.iter_errors(dossier), key=lambda e: list(e.path)):
        location = "/".join(str(part) for part in error.path) or "<root>"
        errors.append(f"{location}: {error.message}")
    return errors


# --------------------------------------------------------------------------
# Deterministic invariants
# --------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_invariants(dossier: Dict[str, Any]) -> List[str]:
    """Re-derive the dossier's numbers and cross-field consistency.

    Every check is a plain comparison against a value the dossier itself
    asserts, so a failure means the response does not describe the assessment
    that was actually performed.
    """
    violations: List[str] = []
    if not isinstance(dossier, dict):
        return ["dossier is not an object"]

    status = dossier.get("qualification_status")
    score = _num(dossier.get("weighted_risk_score"))
    tier = dossier.get("assigned_risk_tier")
    edd = bool(dossier.get("mandatory_edd_triggered"))
    reasons = dossier.get("trigger_reasons") or []
    details = dossier.get("trigger_details") or []

    # -- score / tier ----------------------------------------------------
    if status == "REJECTED_INCOMPLETE":
        if score is not None:
            violations.append(
                "REJECTED_INCOMPLETE must not carry a weighted_risk_score: no "
                f"assessment was performed, but the payload reports {score}"
            )
        if tier is not None:
            violations.append(
                "REJECTED_INCOMPLETE must not carry an assigned_risk_tier"
            )
        if edd:
            violations.append(
                "REJECTED_INCOMPLETE must not report a mandatory EDD trigger: "
                "no trigger was evaluated"
            )
        if dossier.get("guardrail_check_passed"):
            violations.append(
                "guardrail_check_passed must be false on a REJECTED_INCOMPLETE "
                "submission"
            )
    else:
        if score is None:
            violations.append(
                f"status {status} requires a numeric weighted_risk_score"
            )
        else:
            if not (risk_engine.MIN_SCORE <= _dec(score) <= risk_engine.MAX_SCORE):
                violations.append(
                    f"weighted_risk_score {score} is outside the policy range "
                    f"{risk_engine.MIN_SCORE}-{risk_engine.MAX_SCORE}"
                )
            if tier not in risk_engine.TIER_LABELS:
                violations.append(
                    f"assigned_risk_tier {tier!r} is not one of "
                    f"{risk_engine.TIER_LABELS}"
                )
            else:
                # The tier must be the one the policy's band table gives for
                # this score, recomputed here rather than trusted.
                expected_tier, _, _ = risk_engine.tier_for(_dec(score))
                if expected_tier != tier:
                    violations.append(
                        f"score {score} maps to {expected_tier!r} under the "
                        f"policy band table, but the dossier reports {tier!r}"
                    )

        if edd:
            if tier != risk_engine.TIER_HIGH:
                violations.append(
                    f"a mandatory EDD override is reported as {tier!r}; the "
                    f"policy requires {risk_engine.TIER_HIGH!r}"
                )
            if score is not None and score < EDD_FLOOR:
                violations.append(
                    f"a mandatory EDD override must raise the score to at least "
                    f"{EDD_FLOOR}, but the dossier reports {score}"
                )
            if status == "QUALIFIED":
                violations.append(
                    "a mandatory EDD override cannot produce a QUALIFIED status"
                )

    if bool(reasons) != edd:
        violations.append(
            f"trigger_reasons holds {len(reasons)} item(s) but "
            f"mandatory_edd_triggered is {edd}"
        )
    if details and len(details) != len(reasons):
        violations.append(
            f"trigger_details holds {len(details)} item(s) but trigger_reasons "
            f"holds {len(reasons)}"
        )

    # -- Appendix F ------------------------------------------------------
    appendix = dossier.get("appendix_f_score")
    if not isinstance(appendix, dict):
        violations.append("appendix_f_score is missing or is not an object")
    else:
        violations.extend(_check_appendix_f(appendix))

    # -- citations -------------------------------------------------------
    for index, citation in enumerate(dossier.get("rag_retrieval_citations") or []):
        if not isinstance(citation, dict):
            violations.append(f"citation {index} is not an object")
            continue
        if citation.get("verified"):
            # A verified citation is supposed to rest on real retrieved text.
            if not str(citation.get("quote") or "").strip():
                violations.append(
                    f"citation {index} is marked verified but carries no quote"
                )
            if not str(citation.get("source_uri") or "").strip():
                violations.append(
                    f"citation {index} is marked verified but carries no source_uri"
                )

    return violations


def _dec(value: float) -> Any:
    from decimal import Decimal
    return Decimal(str(value))


def _check_appendix_f(appendix: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    total = _num(appendix.get("total_score"))
    components = {
        "financial_standing": _num(appendix.get("financial_standing_score")),
        "technical_capability": _num(appendix.get("technical_capability_score")),
        "quality_hse": _num(appendix.get("quality_hse_score")),
    }

    if total is None:
        violations.append("appendix_f_score.total_score is not a number")
    if any(value is None for value in components.values()):
        violations.append("appendix_f_score is missing a category score")
        return violations

    summed = sum(components.values())  # type: ignore[arg-type]
    if abs(summed - total) > 0.15:
        violations.append(
            f"appendix_f_score components add up to {summed:.1f} but the total "
            f"reports {total:.1f}"
        )
    for name, value in components.items():
        maximum = APPENDIX_F_MAX[name]
        if not (0.0 <= value <= maximum):
            violations.append(
                f"appendix_f_score {name} of {value} is outside 0-{maximum}"
            )

    status = appendix.get("status")
    if status not in ("PASSED", "FAILED"):
        violations.append(
            f"appendix_f_score.status {status!r} is not PASSED or FAILED"
        )
    elif total is not None:
        below_any = any(
            value < maximum * APPENDIX_F_CATEGORY_MINIMUM_RATIO
            for value, maximum in (
                (components["financial_standing"], APPENDIX_F_MAX["financial_standing"]),
                (components["technical_capability"], APPENDIX_F_MAX["technical_capability"]),
                (components["quality_hse"], APPENDIX_F_MAX["quality_hse"]),
            )
        )
        should_pass = total >= APPENDIX_F_PASS_THRESHOLD and not below_any
        if should_pass and status == "FAILED":
            violations.append(
                f"appendix_f_score totals {total} with every category at or "
                f"above its minimum, so the status must be PASSED, not FAILED"
            )
        if not should_pass and status == "PASSED":
            violations.append(
                f"appendix_f_score is marked PASSED but totals {total} against a "
                f"threshold of {APPENDIX_F_PASS_THRESHOLD}"
                + (" with at least one category below its minimum" if below_any else "")
            )
    return violations


def audit(dossier: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and check a dossier, and return the result plus a verdict.

    The dossier is mutated only to downgrade guardrail_check_passed when an
    invariant fails: a compliance claim must never be asserted while the
    arithmetic behind it disagrees.
    """
    schema_errors = validate_dossier(dossier)
    schema_checked = not (len(schema_errors) == 1
                          and schema_errors[0].startswith("schema validation skipped"))
    if not schema_checked:
        schema_errors = []

    invariant_violations = check_invariants(dossier)
    violations = invariant_violations + schema_errors

    if violations:
        dossier["guardrail_check_passed"] = False
        dossier["contract_violations"] = violations

    return {
        "valid": not violations,
        "schema_checked": schema_checked,
        "schema_errors": schema_errors,
        "invariant_violations": invariant_violations,
        "violations": violations,
    }
