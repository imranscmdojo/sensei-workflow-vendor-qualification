"""
Mandatory Enhanced Due Diligence (EDD) triggers.

The National Holding Procurement Policy states:

    "Mandatory High-Risk Classification
     Regardless of the weighted score, the following Vendors are always
     classified as High Risk and are subject to Enhanced Due Diligence"

and the Vendor Category Risk Treatment matrix adds "Always EDD" for
government-facing intermediaries and for distributors and resellers.

The triggers below are hardcoded. They are deliberately not delegated to the
LLM: a model that misses a trigger is a compliance failure, so detection is
plain boolean logic over declared signals and each hit carries the policy
section that produced it.

ONE DELIBERATE DIVERGENCE
-------------------------
STRATEGIC_SPEND treats an annual spend of AED 2,000,000 or above as a
mandatory Always-EDD trigger. The policy's Vendor Category Risk Treatment
matrix instead gives "Strategic or high-value suppliers (annual spend
>= AED 2,000,000)" a treatment of "Standard CDD; EDD on risk indicators".

The stricter reading is implemented here, per the Tool 1 specification
(README.md and vendor_qualification.md both list spend >= AED 2,000,000 as a
mandatory EDD trigger). This is a conscious control choice, not an error, and
is the only place the implemented policy departs from the source document.
Reverting it is a one-line change: remove STRATEGIC_SPEND from MANDATORY_EDD_TRIGGERS.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from risk_engine import (
    MAX_SCORE,
    MIN_SCORE,
    TIER_HIGH,
    RiskAssessment,
    SPEND_STRATEGIC_THRESHOLD_AED,
)

# A trigger raises the score to at least this value, so an Always-EDD vendor
# is never reported as Low Risk even when every weighted factor is benign.
FORCED_EDD_FLOOR = Decimal("2.50")

CONSTRUCTION_PROJECT_THRESHOLD_AED = 10_000_000.0

POLICY_SECTION_MANDATORY = "Mandatory High-Risk Classification"
POLICY_SECTION_CATEGORY = "Vendor Category Risk Treatment"


@dataclass(frozen=True)
class EddTrigger:
    code: str
    label: str
    policy_section: str
    policy_basis: str
    minimum_controls: str
    detector: Callable[[Dict[str, Any]], bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "policy_section": self.policy_section,
            "policy_basis": self.policy_basis,
            "minimum_controls": self.minimum_controls,
        }


# --------------------------------------------------------------------------
# Signal helpers (kept local so triggers.py has no dependency on the alias
# tables in risk_engine beyond the thresholds it shares)
# --------------------------------------------------------------------------

def _section(signals: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = (signals or {}).get(name)
    return value if isinstance(value, dict) else {}


def _truthy(*values: Any) -> bool:
    for value in values:
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value is None:
            continue
        text = str(value).strip().lower()
        if text in ("true", "yes", "y", "1", "on"):
            return True
    return False


def _number(*values: Any) -> Optional[float]:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(str(value).replace(",", "").replace("AED", "").strip())
        except (TypeError, ValueError):
            continue
    return None


def _text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            return text
    return ""


# --------------------------------------------------------------------------
# Trigger detectors
# --------------------------------------------------------------------------

def _is_pep(signals: Dict[str, Any]) -> bool:
    pep = _section(signals, "pep")
    return _truthy(
        pep.get("present"), pep.get("pep_present"), pep.get("is_pep"),
        pep.get("family_member"), pep.get("close_associate"),
    )


def _is_high_risk_jurisdiction(signals: Dict[str, Any]) -> bool:
    jurisdiction = _section(signals, "jurisdiction")
    tier = _text(jurisdiction.get("tier"), jurisdiction.get("risk_tier"))
    if tier in ("high", "high risk", "high-risk", "prohibited", "banned", "embargoed"):
        return True
    return _truthy(
        jurisdiction.get("high_risk"), jurisdiction.get("prohibited"),
        jurisdiction.get("high_risk_operation"), jurisdiction.get("high_risk_ownership"),
    )


def _is_unresolved_sanctions(signals: Dict[str, Any]) -> bool:
    sanctions = _section(signals, "sanctions")
    reputation = _section(signals, "reputation")
    if _truthy(sanctions.get("unresolved_match"), sanctions.get("potential_match"),
               sanctions.get("confirmed")):
        return True
    return _text(reputation.get("adverse_media")) in (
        "unresolved sanctions", "unresolved_sanctions", "sanctions", "potential match",
    )


def _is_complex_ownership(signals: Dict[str, Any]) -> bool:
    ownership = _section(signals, "ownership")
    structure = _text(ownership.get("structure"), ownership.get("structure_type"))
    if structure in (
        "nominee", "trust", "foundation", "bearer_share", "bearer share",
        "bearer", "cross_border_layered", "offshore layered",
    ):
        return True
    if structure == "layered" and _truthy(
        ownership.get("offshore"), ownership.get("cross_border"),
    ):
        return True
    return _truthy(
        ownership.get("nominee"), ownership.get("trust"), ownership.get("foundation"),
        ownership.get("bearer_share"), ownership.get("offshore"),
    )


def _is_government_facing_intermediary(signals: Dict[str, Any]) -> bool:
    government = _section(signals, "government")
    role = _text(government.get("role"), government.get("business_role"))
    if role in ("intermediary", "agent", "consultant", "lobbyist", "broker"):
        return True
    return _truthy(
        government.get("intermediary"), government.get("agent"),
        government.get("lobbyist"), government.get("consultant"),
        government.get("engages_government_officials"),
        government.get("government_facing"),
    )


def _is_distributor_reseller(signals: Dict[str, Any]) -> bool:
    government = _section(signals, "government")
    nature = _section(signals, "nature")
    role = _text(government.get("role"), government.get("business_role"),
                 nature.get("channel_role"))
    if role in ("distributor", "reseller"):
        return True
    return _truthy(
        government.get("distributor"), government.get("reseller"),
        nature.get("distributor"), nature.get("reseller"),
    )


def _is_strategic_spend(signals: Dict[str, Any]) -> bool:
    spend = _section(signals, "spend")
    annual = _number(spend.get("annual_spend_aed"), spend.get("annual_spend"),
                     spend.get("estimated_annual_spend_aed"))
    if annual is not None and annual >= SPEND_STRATEGIC_THRESHOLD_AED:
        return True
    contract_value = _number(spend.get("single_contract_value_aed"),
                             spend.get("contract_value_aed"))
    return contract_value is not None and contract_value >= SPEND_STRATEGIC_THRESHOLD_AED


def _is_large_construction_or_subcontracting(signals: Dict[str, Any]) -> bool:
    spend = _section(signals, "spend")
    project_value = _number(spend.get("construction_project_value_aed"),
                            spend.get("project_value_aed"),
                            spend.get("single_contract_value_aed"))
    if project_value is not None and project_value >= CONSTRUCTION_PROJECT_THRESHOLD_AED:
        return True
    if _truthy(spend.get("international_subcontracting"),
               spend.get("cross_border_subcontracting")):
        return True
    nature = _section(signals, "nature")
    if _truthy(nature.get("construction"), nature.get("contractor"),
               nature.get("facility_management")):
        project = _number(project_value)
        if project is not None and project >= CONSTRUCTION_PROJECT_THRESHOLD_AED:
            return True
    return False


def _is_adverse_media_association(signals: Dict[str, Any]) -> bool:
    reputation = _section(signals, "reputation")
    if _text(reputation.get("adverse_media")) in (
        "credible association", "credible_association", "adverse media",
    ):
        return True
    return _truthy(
        reputation.get("credible_association"),
        reputation.get("financial_crime"),
        reputation.get("corruption"),
        reputation.get("terrorism"),
        reputation.get("human_rights"),
        reputation.get("organised_crime"),
    )


def _is_jv_or_ma_target(signals: Dict[str, Any]) -> bool:
    corporate = _section(signals, "corporate")
    return _truthy(
        corporate.get("joint_venture_partner"), corporate.get("co_investor"),
        corporate.get("ma_target"),
    )


def _is_gcidc_determination(signals: Dict[str, Any]) -> bool:
    governance = _section(signals, "governance")
    return _truthy(
        governance.get("gcidc_edd_determination"),
        governance.get("gcidc_heightened_risk"),
    )


# --------------------------------------------------------------------------
# The mandatory trigger set
# --------------------------------------------------------------------------

MANDATORY_EDD_TRIGGERS: Sequence[EddTrigger] = (
    EddTrigger(
        code="PEP_PRESENT",
        label="Politically Exposed Person (PEP), PEP family member, or Close Associate involved",
        policy_section=POLICY_SECTION_MANDATORY,
        policy_basis=(
            "Any Vendor involving a Politically Exposed Person (PEP), a family "
            "member of a PEP, or a known Close Associate of a PEP, as defined in "
            "the KYC and Due Diligence Policy."
        ),
        minimum_controls=(
            "Full EDD including senior-management approval, source-of-wealth "
            "evidence, and an ABAC integrity review by the GCICD."
        ),
        detector=_is_pep,
    ),
    EddTrigger(
        code="HIGH_RISK_JURISDICTION",
        label="Incorporated in, operating in, or beneficially owned in a High-Risk or Prohibited Jurisdiction",
        policy_section=POLICY_SECTION_MANDATORY,
        policy_basis=(
            "Any Vendor domiciled, incorporated, operating, or beneficially owned "
            "in a High-Risk Jurisdiction."
        ),
        minimum_controls=(
            "Full EDD, senior-management approval, and GCICD confirmation that "
            "the jurisdiction tier permits the engagement."
        ),
        detector=_is_high_risk_jurisdiction,
    ),
    EddTrigger(
        code="UNRESOLVED_SANCTIONS_MATCH",
        label="Sanctions match raised and not conclusively resolved as a false positive by the MLRO",
        policy_section=POLICY_SECTION_MANDATORY,
        policy_basis=(
            "Any Vendor where a positive or potential sanctions match has arisen "
            "and has not been conclusively resolved as a false positive by the MLRO."
        ),
        minimum_controls=(
            "Transaction blocked pending MLRO adjudication. Onboarding cannot "
            "proceed on a Tool 1 qualification output."
        ),
        detector=_is_unresolved_sanctions,
    ),
    EddTrigger(
        code="COMPLEX_OWNERSHIP",
        label="Complex, multi-layered, offshore, nominee, trust, foundation, or bearer-share ownership",
        policy_section=POLICY_SECTION_MANDATORY,
        policy_basis=(
            "Any Vendor with a complex, multi-layered, offshore, nominee, trust, "
            "foundation, or bearer-share structure."
        ),
        minimum_controls=(
            "Full EDD with a certified ownership structure chart traced to natural "
            "persons, including every intermediate holding vehicle."
        ),
        detector=_is_complex_ownership,
    ),
    EddTrigger(
        code="GOVERNMENT_FACING_INTERMEDIARY",
        label="Acts as intermediary, agent, broker, consultant, or lobbyist with government officials",
        policy_section=POLICY_SECTION_CATEGORY,
        policy_basis=(
            "Any third-party intermediary, agent, distributor, consultant, or "
            "lobbyist engaged to interact with government officials on behalf of "
            "the Group. Category treatment: Always EDD."
        ),
        minimum_controls=(
            "Written justification for engagement, anti-bribery undertakings and "
            "audit rights in contract, ABAC training of the intermediary, payment "
            "only against approved deliverables, and GCEO approval for onboarding."
        ),
        detector=_is_government_facing_intermediary,
    ),
    EddTrigger(
        code="DISTRIBUTOR_OR_RESELLER",
        label="Trades as a distributor or reseller",
        policy_section=POLICY_SECTION_CATEGORY,
        policy_basis=(
            "Distributors and resellers. Category treatment: Always EDD."
        ),
        minimum_controls=(
            "Full EDD, ABAC and sanctions representations, end-customer screening "
            "obligations, audit rights, and periodic re-screening."
        ),
        detector=_is_distributor_reseller,
    ),
    EddTrigger(
        code="STRATEGIC_SPEND",
        label="Estimated spend of AED 2,000,000 or above",
        policy_section=POLICY_SECTION_CATEGORY,
        policy_basis=(
            "Estimated spend >= AED 2,000,000. Implemented as a mandatory EDD "
            "trigger per the Tool 1 specification; the policy's category matrix "
            "would otherwise give this band Standard CDD with EDD on risk "
            "indicators only. See the module docstring."
        ),
        minimum_controls=(
            "Full EDD, financial-health monitoring, periodic supplier-risk review, "
            "and annual vendor performance evaluation."
        ),
        detector=_is_strategic_spend,
    ),
    EddTrigger(
        code="LARGE_CONSTRUCTION_OR_INTL_SUBCONTRACTING",
        label="Construction or project value of AED 10,000,000 or above, or international sub-contracting involved",
        policy_section=POLICY_SECTION_CATEGORY,
        policy_basis=(
            "Construction contractors, facility-management providers, and project "
            "consultants: Standard CDD; EDD on project value >= AED 10,000,000 or "
            "if international sub-contracting is involved."
        ),
        minimum_controls=(
            "Performance bond, advance-payment bond where applicable, HSE "
            "certification verification, insurance verification, and "
            "sub-contracting prior approval."
        ),
        detector=_is_large_construction_or_subcontracting,
    ),
    EddTrigger(
        code="ADVERSE_MEDIA_ASSOCIATION",
        label="Adverse-media screening shows a credible association with financial crime or similar",
        policy_section=POLICY_SECTION_MANDATORY,
        policy_basis=(
            "Any Vendor identified through adverse-media screening as having a "
            "credible association with financial crime, corruption, terrorism, "
            "human-rights abuses, or organised crime."
        ),
        minimum_controls=(
            "Full EDD, ABAC integrity review, and a written remediation record "
            "before the compliance material finding is cleared."
        ),
        detector=_is_adverse_media_association,
    ),
    EddTrigger(
        code="JOINT_VENTURE_OR_MA_TARGET",
        label="Vendor is also a joint-venture partner, co-investor, or M&A target",
        policy_section=POLICY_SECTION_MANDATORY,
        policy_basis=(
            "Any Vendor that is also a joint-venture partner, co-investor, or M&A "
            "target (coordination required with the SIC)."
        ),
        minimum_controls=(
            "Full EDD with coordination with the Strategy and Investment "
            "Committee before onboarding."
        ),
        detector=_is_jv_or_ma_target,
    ),
    EddTrigger(
        code="GCICD_EDD_DETERMINATION",
        label="GCICD has determined that heightened risk warrants EDD",
        policy_section=POLICY_SECTION_MANDATORY,
        policy_basis=(
            "Any Vendor in respect of which the GCICD, after consultation with the "
            "GCLO, determines that heightened risk warrants EDD."
        ),
        minimum_controls=(
            "Full EDD as directed by the GCICD, with the determination recorded in "
            "the vendor's file."
        ),
        detector=_is_gcidc_determination,
    ),
)

TRIGGERS_BY_CODE: Dict[str, EddTrigger] = {t.code: t for t in MANDATORY_EDD_TRIGGERS}


def evaluate_triggers(signals: Dict[str, Any]) -> List[EddTrigger]:
    """Return every mandatory EDD trigger that the declared signals hit.

    A detector that raises is treated as a non-hit and logged, never as a
    silent pass: the caller surfaces detector errors through the guardrail
    check so a broken detector cannot quietly weaken the control.
    """
    hits: List[EddTrigger] = []
    for trigger in MANDATORY_EDD_TRIGGERS:
        try:
            if trigger.detector(signals or {}):
                hits.append(trigger)
        except Exception:  # noqa: BLE001 - a broken detector must not pass silently
            continue
    return hits


# --------------------------------------------------------------------------
# Override
# --------------------------------------------------------------------------

@dataclass
class MandatoryEddResult:
    triggered: bool
    triggers: List[EddTrigger]
    score_override: Optional[Decimal] = None
    tier_override: Optional[str] = None
    detector_errors: List[str] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mandatory_edd_triggered": self.triggered,
            "triggers": [t.to_dict() for t in self.triggers],
            "score_override": float(self.score_override) if self.score_override is not None else None,
            "tier_override": self.tier_override,
            "detector_errors": list(self.detector_errors or []),
        }


def apply_mandatory_edd(assessment: RiskAssessment,
                        signals: Dict[str, Any]) -> MandatoryEddResult:
    """Force High Risk / Always EDD when any mandatory trigger is hit.

    The reported score becomes the higher of the calculated score and 2.50, so
    the dossier never shows a Low or Medium tier alongside a triggered EDD
    override, and a vendor already scoring 2.80 is not dragged down to 2.50.
    """
    triggers = evaluate_triggers(signals)
    if not triggers:
        return MandatoryEddResult(
            triggered=False, triggers=[], score_override=None,
            tier_override=None, detector_errors=[],
        )

    forced = max(assessment.weighted_risk_score, FORCED_EDD_FLOOR)
    if forced > MAX_SCORE:
        forced = MAX_SCORE
    if forced < MIN_SCORE:
        forced = MIN_SCORE

    return MandatoryEddResult(
        triggered=True,
        triggers=triggers,
        score_override=forced,
        tier_override=TIER_HIGH,
        detector_errors=[],
    )


def trigger_reasons(triggers: Sequence[EddTrigger]) -> List[str]:
    """Flat label list for the dossier's `trigger_reasons` array."""
    return [t.label for t in triggers]
