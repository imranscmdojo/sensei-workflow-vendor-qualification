"""
Deterministic Weighted Risk Score engine — Vendor Qualification & Risk Tiering.

This module is the single source of truth for every number in the qualification
dossier. It is pure Python: no LLM, no RAG, no network, no clock. The Gemini
agent may add narrative, policy citations and Appendix F rationale, but the
weighted risk score, the risk tier, the due-diligence level and the refresh
cycle are computed here and can never be overridden by model output.

Policy basis
------------
National Holding Procurement Policy, "VENDOR LIFECYCLE MANAGEMENT":

    "Every Vendor is assigned a risk score based on the Risk Scoring Matrix of
     the KYC and Due Diligence Policy (Appendix B thereof)."

    Weighted Risk Score | Risk Level | Due Diligence        | Refresh Cycle
    1.00 - 1.60         | Low        | Simplified (SDD)     | Every 3 years
    1.61 - 2.20         | Medium     | Standard (CDD)       | Every 2 years
    2.21 - 3.00         | High       | Enhanced (EDD)       | Annually

    "The risk score is calculated from the following weighted factors":
      1. ownership and control structure
      2. jurisdiction of incorporation and principal operations
      3. nature and sensitivity of the goods or services being procured
      4. annual spend and strategic importance
      5. exposure to government interaction or licensing
      6. proposed payment structure (success fees, commissions, unusual)
      7. reputational indicators from adverse-media screening
      8. prior relationship history with the Group

WEIGHTS
-------
The policy states the weights are "as published by the GCICD" in the KYC and
Due Diligence Policy (Appendix B), which is not part of the ingested corpus.
The weights below are therefore documented defaults chosen to reflect how much
influence each factor carries in the policy's own narrative, and they are the
single place to change them when the GCICD matrix is supplied.

Jurisdiction and ownership are the two factors the policy singles out for
mandatory treatment, so they carry the largest weights. Payment structure and
government exposure are weighted equally because the policy pairs them
(constructive fraud) and drives both into the ABAC integrity review. Spend is
kept below those four but above the residual factors, which are modifiers
rather than drivers. The weights sum to exactly 1.00, asserted at import time
and covered by a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Factor weights — sum to 1.00 (asserted below)
# --------------------------------------------------------------------------

FACTOR_WEIGHTS: Dict[str, float] = {
    "jurisdiction": 0.25,
    "ownership_control": 0.20,
    "government_exposure": 0.15,
    "payment_structure": 0.15,
    "annual_spend": 0.10,
    "nature_sensitivity": 0.05,
    "adverse_media": 0.05,
    "prior_relationship": 0.05,
}

assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 1e-9, "Risk factor weights must sum to 1.00"

MIN_SCORE = Decimal("1.00")
MAX_SCORE = Decimal("3.00")
QUANTUM = Decimal("0.01")

POLICY_DOCUMENT = "Procurement Policy_Updated_2.pdf"
POLICY_SECTION_MATRIX = "KYC and Due Diligence (SDD / CDD / EDD) - Risk Scoring Matrix"
POLICY_SECTION_FACTORS = "Risk Factors"
POLICY_SECTION_CATEGORY = "Vendor Category Risk Treatment"

# --------------------------------------------------------------------------
# Tier bands — verbatim from the policy's Risk Scoring Matrix. Thresholds are
# never rounded, relaxed, or extended. Band edges are inclusive-upper.
# --------------------------------------------------------------------------

TIER_BANDS: Tuple[Tuple[Decimal, str, str, str], ...] = (
    (Decimal("1.60"), "Low Risk (SDD)", "Simplified Due Diligence (SDD)", "Every 3 Years"),
    (Decimal("2.20"), "Medium Risk (CDD)", "Standard Customer Due Diligence (CDD)", "Every 2 Years"),
    (MAX_SCORE, "High Risk (EDD)", "Enhanced Due Diligence (EDD)", "Annual Refresh"),
)

TIER_LABELS = tuple(band[1] for band in TIER_BANDS)

# Aliases so callers can use the shorter key without duplicating the literals.
TIER_LOW = "Low Risk (SDD)"
TIER_MEDIUM = "Medium Risk (CDD)"
TIER_HIGH = "High Risk (EDD)"

# Sentinels used when a factor has not been signalled at all. A factor the
# form did not ask about is *unknown*, never "clean" — unknown sits above the
# floor so an unanswered question can never manufacture a Low Risk vendor.
SCORE_UNKNOWN = Decimal("2.00")
SCORE_FLOOR = Decimal("1.00")


# --------------------------------------------------------------------------
# Factor scoring rules
# --------------------------------------------------------------------------
#
# Each entry maps a normalized signal value to (score, human-readable rule,
# policy basis). Scores are anchored on 1.00 / 1.50 / 2.00 / 2.50 / 3.00.
# The default is used when the signal is absent or unrecognised, so a new form
# field can never crash or silently pass through as Low Risk.

UNKNOWN_RULE = (
    "Not declared on the submission. An undeclared factor cannot be scored as "
    "minimum risk, so it is assessed at the neutral 2.00 anchor pending "
    "confirmation."
)


@dataclass(frozen=True)
class FactorRule:
    score: Decimal
    rule: str
    basis: str


def _rules(pairs: Dict[Any, Tuple[str, str]]) -> Dict[Any, FactorRule]:
    return {
        key: FactorRule(Decimal(str(score)), rule, basis)
        for key, (score, rule, basis) in pairs.items()
    }


# 1. Ownership and control structure ----------------------------------------
OWNERSHIP_RULES = _rules({
    "simple": ("1.00",
               "Ownership is direct and transparent: named individuals hold clear "
               "voting rights and there is no intermediate holding vehicle.",
               POLICY_SECTION_FACTORS),
    "domestic_layered": ("1.50",
                         "A simple parent plus one domestic intermediate holding "
                         "entity, with ownership still traceable to natural persons.",
                         POLICY_SECTION_FACTORS),
    "layered": ("2.00",
                "Multi-layer ownership requiring tracing through more than one "
                "holding entity before a natural person is reached.",
                POLICY_SECTION_FACTORS),
    "cross_border_layered": ("2.50",
                             "Multi-layer ownership crossing a jurisdiction "
                             "boundary, so beneficial ownership must be traced "
                             "through at least two legal systems.",
                             POLICY_SECTION_FACTORS),
    "nominee": ("3.00",
                "Nominee or undisclosed-shareholder arrangements, where the "
                "registered holder may not be the beneficial owner.",
                POLICY_SECTION_FACTORS),
    "trust": ("3.00",
              "Trust, foundation, or private-benefit-company ownership, where "
              "control is exercised by trustees rather than by identified owners.",
              POLICY_SECTION_FACTORS),
    "bearer_share": ("3.00",
                     "Bearer shares with no registered holder, so beneficial "
                     "ownership cannot be established from company records.",
                     POLICY_SECTION_FACTORS),
})

# 2. Jurisdiction of incorporation and principal operations -----------------
JURISDICTION_RULES = _rules({
    "low": ("1.00",
            "Incorporated and operating in a Low-risk jurisdiction, with no "
            "material operations in a higher-risk jurisdiction.",
            POLICY_SECTION_FACTORS),
    "medium": ("2.00",
               "Incorporated in, or operating materially through, a Medium-risk "
               "jurisdiction.",
               POLICY_SECTION_FACTORS),
    "high": ("3.00",
             "Incorporated in, operating in, or beneficially owned in a "
             "High-risk jurisdiction.",
             POLICY_SECTION_FACTORS),
    "prohibited": ("3.00",
                   "Jurisdiction is Prohibited for the Group; engagement is not "
                   "permitted irrespective of the weighted score.",
                   POLICY_SECTION_FACTORS),
})

# 3. Nature and sensitivity of the goods or services -------------------------
NATURE_RULES = _rules({
    "routine": ("1.00",
                "Routine, non-sensitive goods or services with no regulatory, "
                "defence, or critical-infrastructure dependency.",
                POLICY_SECTION_FACTORS),
    "moderate": ("1.50",
                 "Standard goods or services with a moderate substitution "
                 "difficulty or limited operational dependency.",
                 POLICY_SECTION_FACTORS),
    "sensitive": ("2.50",
                  "Sensitive goods or services, or a supply with material "
                  "operational dependency on a single vendor.",
                  POLICY_SECTION_FACTORS),
    "critical": ("3.00",
                 "Critical or highly sensitive goods or services, or a category "
                 "carrying regulatory or national-security sensitivity.",
                 POLICY_SECTION_FACTORS),
    "financial_institution": ("2.00",
                              "Financial-institution counterparty: additional "
                              "regulatory-status verification applies rather "
                              "than a simple scale uplift.",
                              POLICY_SECTION_CATEGORY),
})

# 4. Annual spend and strategic importance ----------------------------------
#    AED 2,000,000 is the policy's Strategic / high-value threshold. It is also
#    a mandatory Always-EDD trigger in triggers.py (STRATEGIC_SPEND), which is
#    deliberately stricter than the policy's category treatment; see that file.
SPEND_STRATEGIC_THRESHOLD_AED = 2_000_000.0
SPEND_ELEVATED_THRESHOLD_AED = 500_000.0

SPEND_RULES = _rules({
    "under_100k": ("1.00",
                   "Annual spend below AED 100,000, within the small-supply "
                   "exception band.",
                   POLICY_SECTION_CATEGORY),
    "under_500k": ("1.50",
                   "Annual spend between AED 100,000 and AED 500,000.",
                   POLICY_SECTION_CATEGORY),
    "under_2m": ("2.00",
                 "Annual spend between AED 500,000 and AED 2,000,000.",
                 POLICY_SECTION_CATEGORY),
    "strategic_2m_plus": ("2.50",
                          "Strategic or high-value supplier at an annual spend "
                          "of AED 2,000,000 or above.",
                          POLICY_SECTION_CATEGORY),
    "strategic_10m_plus": ("3.00",
                           "Very large annual commitment, or a single contract "
                           "value that would dominate the supplier's capacity.",
                           POLICY_SECTION_CATEGORY),
})

# 5. Exposure to government interaction or licensing ------------------------
GOVERNMENT_RULES = _rules({
    "none": ("1.00",
             "No interaction with government officials and no licensing or "
             "permitting dependency in the delivery model.",
             POLICY_SECTION_FACTORS),
    "indirect": ("1.50",
                 "Incidental public-sector exposure through permits, customs, or "
                 "utility interfaces, without a direct government counterparty.",
                 POLICY_SECTION_FACTORS),
    "direct": ("2.50",
               "Direct engagement with government officials or entities in the "
               "ordinary course of delivery, including licensing responsibilities.",
               POLICY_SECTION_FACTORS),
    "government_facing": ("3.00",
                          "The vendor acts as intermediary, agent, distributor, "
                          "reseller, broker, consultant, or lobbyist in relation "
                          "to government officials.",
                          POLICY_SECTION_CATEGORY),
})

# 6. Proposed payment structure --------------------------------------------
PAYMENT_RULES = _rules({
    "standard": ("1.00",
                 "Standard invoicing against accepted deliverables, settled to "
                 "an account in the vendor's own name in its country of "
                 "incorporation.",
                 POLICY_SECTION_FACTORS),
    "extended_terms": ("1.25",
                       "Non-standard but ordinary payment terms, such as extended "
                       "credit or staged payments against milestones.",
                       POLICY_SECTION_FACTORS),
    "third_party": ("2.50",
                    "Payment routed to a third party who is not the contracting "
                    "vendor, requiring the beneficial beneficiary to be identified.",
                    POLICY_SECTION_FACTORS),
    "foreign_account": ("2.75",
                        "Payment into an account held outside the vendor's "
                        "country of incorporation.",
                        POLICY_SECTION_FACTORS),
    "success_fee_or_commission": ("3.00",
                                  "Success fees, commissions, or other unusual "
                                  "remuneration that creates an incentive to act "
                                  "improperly to secure the outcome.",
                                  POLICY_SECTION_FACTORS),
    "cash": ("3.00",
             "Cash payment requested, which the policy expects to be exceptional "
             "and supported by a documented commercial reason.",
             POLICY_SECTION_FACTORS),
})

# 7. Reputational indicators from adverse-media screening -------------------
REPUTATION_RULES = _rules({
    "clear": ("1.00",
              "Adverse-media and public-records screening returned no material "
              "indicator.",
              POLICY_SECTION_FACTORS),
    "minor": ("1.50",
              "Only immaterial or historic public reporting, with no credible "
              "association with financial crime.",
              POLICY_SECTION_FACTORS),
    "regulatory": ("2.50",
                   "Regulatory enforcement, licence suspension, or tax-evasion "
                   "allegation against the vendor, its directors, or its "
                   "controlling owners within the last five years.",
                   POLICY_SECTION_FACTORS),
    "credible_association": ("3.00",
                             "A credible association with financial crime, "
                             "corruption, terrorism, human-rights abuses, or "
                             "organised crime.",
                             POLICY_SECTION_FACTORS),
    "unresolved_sanctions": ("3.00",
                             "A positive or potential sanctions match that the "
                             "MLRO has not conclusively resolved as a false "
                             "positive.",
                             POLICY_SECTION_FACTORS),
})

# 8. Prior relationship history with the Group ------------------------------
RELATIONSHIP_RULES = _rules({
    "first_time": ("1.00",
                   "No prior relationship with the Group; this is a new vendor.",
                   POLICY_SECTION_FACTORS),
    "prior_qualified": ("1.00",
                        "Previously qualified vendor with a clean compliance and "
                        "performance record.",
                        POLICY_SECTION_FACTORS),
    "prior_issues": ("2.50",
                     "Prior relationship with recorded compliance, performance, "
                     "or integrity issues that were closed out.",
                     POLICY_SECTION_FACTORS),
    "prior_suspended": ("3.00",
                        "Previously suspended or blacklisted vendor. The policy "
                        "requires a 24-month remediation period and fresh "
                        "prequalification before reinstatement.",
                        POLICY_SECTION_FACTORS),
})


# --------------------------------------------------------------------------
# Signal normalisation
# --------------------------------------------------------------------------
#
# The wizard form and the RAG layer both feed the same signal bag. Values are
# normalised here so free-text from a parsed form ("Layered - 2 offshore SPVs")
# and structured input ("layered") land on the same rule.

def _norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    return text or None


def _first(*values: Any) -> Optional[str]:
    for value in values:
        normalised = _norm(value)
        if normalised is not None:
            return normalised
    return None


def _as_bool(*values: Any) -> bool:
    for value in values:
        normalised = _norm(value)
        if normalised is not None:
            return normalised in ("true", "yes", "y", "1", "on")
    return False


def _as_float(*values: Any) -> Optional[float]:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(str(value).replace(",", "").replace("AED", "").strip())
        except (TypeError, ValueError):
            continue
    return None


# Ownership aliases: a parsed form will describe the structure in prose.
OWNERSHIP_ALIASES: Dict[str, str] = {
    "simple": "simple",
    "direct": "simple",
    "transparent": "simple",
    "individual": "simple",
    "sole proprietor": "simple",
    "single shareholder": "simple",
    "domestic": "domestic_layered",
    "domestic_layered": "domestic_layered",
    "layered": "layered",
    "multi-layered": "layered",
    "multi_layered": "layered",
    "complex": "layered",
    "holdco": "layered",
    "cross_border": "cross_border_layered",
    "cross-border": "cross_border_layered",
    "cross_border_layered": "cross_border_layered",
    "offshore": "cross_border_layered",
    "offshore layered": "cross_border_layered",
    "nominee": "nominee",
    "nominee shareholder": "nominee",
    "trust": "trust",
    "foundation": "trust",
    "private benefit company": "trust",
    "pbc": "trust",
    "bearer": "bearer_share",
    "bearer share": "bearer_share",
    "bearer shares": "bearer_share",
}

JURISDICTION_ALIASES: Dict[str, str] = {
    "low": "low",
    "low risk": "low",
    "standard": "low",
    "medium": "medium",
    "medium risk": "medium",
    "moderate": "medium",
    "high": "high",
    "high risk": "high",
    "high-risk": "high",
    "prohibited": "prohibited",
    "banned": "prohibited",
    "embargoed": "prohibited",
}

NATURE_ALIASES: Dict[str, str] = {
    "routine": "routine",
    "non-sensitive": "routine",
    "non sensitive": "routine",
    "standard goods": "routine",
    "moderate": "moderate",
    "standard": "moderate",
    "sensitive": "sensitive",
    "critical": "critical",
    "highly sensitive": "critical",
    "financial institution": "financial_institution",
    "financial_institution": "financial_institution",
    "bank": "financial_institution",
    "custodian": "financial_institution",
    "insurer": "financial_institution",
}

GOVERNMENT_ROLE_ALIASES: Dict[str, str] = {
    "none": "none",
    "no": "none",
    "": "none",
    "direct supplier": "none",
    "incidental": "indirect",
    "indirect": "indirect",
    "permits": "indirect",
    "customs": "indirect",
    "direct": "direct",
    "licensing": "direct",
    "government official": "direct",
    "public sector": "direct",
    "government_facing": "government_facing",
    "government-facing": "government_facing",
    "intermediary": "government_facing",
    "agent": "government_facing",
    "distributor": "government_facing",
    "reseller": "government_facing",
    "broker": "government_facing",
    "consultant": "government_facing",
    "lobbyist": "government_facing",
}

PAYMENT_ALIASES: Dict[str, str] = {
    "standard": "standard",
    "standard terms": "standard",
    "normal": "standard",
    "invoice": "standard",
    "extended": "extended_terms",
    "extended terms": "extended_terms",
    "staged": "extended_terms",
    "milestone": "extended_terms",
    "third_party": "third_party",
    "third party": "third_party",
    "third-party": "third_party",
    "foreign_account": "foreign_account",
    "foreign account": "foreign_account",
    "offshore account": "foreign_account",
    "success_fee": "success_fee_or_commission",
    "success fee": "success_fee_or_commission",
    "commission": "success_fee_or_commission",
    "success fee or commission": "success_fee_or_commission",
    "unusual remuneration": "success_fee_or_commission",
    "cash": "cash",
    "cash payment": "cash",
}

REPUTATION_ALIASES: Dict[str, str] = {
    "clear": "clear",
    "none": "clear",
    "clean": "clear",
    "no findings": "clear",
    "minor": "minor",
    "immaterial": "minor",
    "historic": "minor",
    "regulatory": "regulatory",
    "enforcement": "regulatory",
    "licence suspension": "regulatory",
    "tax evasion": "regulatory",
    "credible association": "credible_association",
    "credible_association": "credible_association",
    "adverse media": "credible_association",
    "unresolved sanctions": "unresolved_sanctions",
    "unresolved_sanctions": "unresolved_sanctions",
    "sanctions": "unresolved_sanctions",
    "potential match": "unresolved_sanctions",
}

RELATIONSHIP_ALIASES: Dict[str, str] = {
    "first_time": "first_time",
    "first time": "first_time",
    "new": "first_time",
    "none": "first_time",
    "prior_qualified": "prior_qualified",
    "existing": "prior_qualified",
    "incumbent": "prior_qualified",
    "prior issues": "prior_issues",
    "prior_issues": "prior_issues",
    "issues": "prior_issues",
    "prior_suspended": "prior_suspended",
    "suspended": "prior_suspended",
    "blacklisted": "prior_suspended",
    "terminated": "prior_suspended",
}


def _alias(value: Optional[str], table: Dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    if value in table:
        return table[value]
    collapsed = value.replace("  ", " ").strip()
    if collapsed in table:
        return table[collapsed]
    for key, mapped in table.items():
        if key in collapsed:
            return mapped
    return None


# --------------------------------------------------------------------------
# Spend banding
# --------------------------------------------------------------------------

def spend_band(annual_spend_aed: Optional[float],
               single_contract_value_aed: Optional[float] = None) -> str:
    """Map spend to a band. The larger of annual spend and a single contract
    value is used, because a large one-off commitment carries the same
    dependency risk as recurring spend of the same size."""
    values = [v for v in (annual_spend_aed, single_contract_value_aed) if v is not None]
    if not values:
        return "unknown"
    peak = max(values)
    if peak >= 10_000_000:
        return "strategic_10m_plus"
    if peak >= SPEND_STRATEGIC_THRESHOLD_AED:
        return "strategic_2m_plus"
    if peak >= SPEND_ELEVATED_THRESHOLD_AED:
        return "under_2m"
    if peak >= 100_000:
        return "under_500k"
    return "under_100k"


# --------------------------------------------------------------------------
# Factor scoring
# --------------------------------------------------------------------------

@dataclass
class FactorResult:
    factor: str
    label: str
    weight: float
    score: Decimal
    rule: str
    basis: str
    signal: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "label": self.label,
            "weight": self.weight,
            "score": float(self.score),
            "rule": self.rule,
            "basis": self.basis,
            "signal": self.signal,
        }


FACTOR_LABELS: Dict[str, str] = {
    "jurisdiction": "Jurisdiction of Incorporation and Principal Operations",
    "ownership_control": "Ownership and Control Structure",
    "government_exposure": "Exposure to Government Interaction or Licensing",
    "payment_structure": "Proposed Payment Structure",
    "annual_spend": "Annual Spend and Strategic Importance",
    "nature_sensitivity": "Nature and Sensitivity of Goods or Services",
    "adverse_media": "Reputational Indicators from Adverse-Media Screening",
    "prior_relationship": "Prior Relationship History with the Group",
}

UNKNOWN_FACTOR = FactorRule(SCORE_UNKNOWN, UNKNOWN_RULE, POLICY_SECTION_FACTORS)


def _resolve(rules: Dict[str, FactorRule], value: Optional[str]) -> FactorRule:
    if value is None:
        return UNKNOWN_FACTOR
    return rules.get(value, UNKNOWN_FACTOR)


def score_factors(signals: Dict[str, Any]) -> List[FactorResult]:
    """Score all eight weighted factors from a normalised signal bag.

    Every factor always returns a result. A factor with no usable signal is
    scored at the neutral 2.00 anchor and flagged, so incompleteness surfaces
    as mid-range risk instead of silently as Low Risk.
    """
    signals = signals or {}

    ownership = signals.get("ownership") or {}
    jurisdiction = signals.get("jurisdiction") or {}
    government = signals.get("government") or {}
    payment = signals.get("payment") or {}
    spend = signals.get("spend") or {}
    nature = signals.get("nature") or {}
    reputation = signals.get("reputation") or {}
    relationship = signals.get("relationship") or {}

    # -- ownership ---------------------------------------------------------
    ownership_signal = _alias(
        _first(ownership.get("structure"), ownership.get("structure_type")),
        OWNERSHIP_ALIASES,
    )
    if ownership_signal is None:
        if _as_bool(ownership.get("trust"), ownership.get("foundation"),
                    ownership.get("bearer_share")):
            ownership_signal = "trust"
        elif _as_bool(ownership.get("nominee")):
            ownership_signal = "nominee"
        elif _as_bool(ownership.get("offshore"), ownership.get("cross_border")):
            ownership_signal = "cross_border_layered"
        elif _as_bool(ownership.get("layered"), ownership.get("multi_layered")):
            ownership_signal = "layered"
    ownership_rule = _resolve(OWNERSHIP_RULES, ownership_signal)

    # -- jurisdiction ------------------------------------------------------
    declared_tier = _first(jurisdiction.get("tier"), jurisdiction.get("risk_tier"))
    jurisdiction_signal = _alias(declared_tier, JURISDICTION_ALIASES)
    if jurisdiction_signal is None:
        if _as_bool(jurisdiction.get("prohibited")):
            jurisdiction_signal = "prohibited"
        elif _as_bool(jurisdiction.get("high_risk")):
            jurisdiction_signal = "high"
        elif _as_bool(jurisdiction.get("medium_risk")):
            jurisdiction_signal = "medium"
        elif _as_bool(jurisdiction.get("low_risk")):
            jurisdiction_signal = "low"
    jurisdiction_rule = _resolve(JURISDICTION_RULES, jurisdiction_signal)

    # -- nature / sensitivity ---------------------------------------------
    nature_signal = _alias(
        _first(nature.get("sensitivity"), nature.get("nature"), nature.get("category")),
        NATURE_ALIASES,
    )
    if nature_signal is None:
        if _as_bool(nature.get("financial_institution")):
            nature_signal = "financial_institution"
        elif _as_bool(nature.get("critical")):
            nature_signal = "critical"
        elif _as_bool(nature.get("sensitive")):
            nature_signal = "sensitive"
    nature_rule = _resolve(NATURE_RULES, nature_signal)

    # -- annual spend ------------------------------------------------------
    spend_signal = spend.get("band") or spend_band(
        _as_float(spend.get("annual_spend_aed"), spend.get("annual_spend")),
        _as_float(spend.get("single_contract_value_aed"),
                  spend.get("contract_value_aed")),
    )
    spend_signal = _norm(spend_signal)
    if spend_signal not in SPEND_RULES:
        spend_signal = None
    spend_rule = _resolve(SPEND_RULES, spend_signal)

    # -- government exposure ----------------------------------------------
    government_signal = _alias(
        _first(government.get("role"), government.get("business_role")),
        GOVERNMENT_ROLE_ALIASES,
    )
    if government_signal is None:
        if _as_bool(government.get("government_facing"), government.get("intermediary"),
                    government.get("lobbyist"), government.get("distributor"),
                    government.get("reseller"), government.get("broker")):
            government_signal = "government_facing"
        elif _as_bool(government.get("direct_government_exposure")):
            government_signal = "direct"
        elif _as_bool(government.get("indirect_government_exposure")):
            government_signal = "indirect"
    government_rule = _resolve(GOVERNMENT_RULES, government_signal)

    # -- payment structure -------------------------------------------------
    payment_signal = _alias(
        _first(payment.get("structure"), payment.get("payment_structure")),
        PAYMENT_ALIASES,
    )
    if payment_signal is None:
        if _as_bool(payment.get("success_fee"), payment.get("commission")):
            payment_signal = "success_fee_or_commission"
        elif _as_bool(payment.get("cash")):
            payment_signal = "cash"
        elif _as_bool(payment.get("foreign_account")):
            payment_signal = "foreign_account"
        elif _as_bool(payment.get("third_party")):
            payment_signal = "third_party"
    payment_rule = _resolve(PAYMENT_RULES, payment_signal)

    # -- reputational indicators ------------------------------------------
    reputation_signal = _alias(
        _first(reputation.get("adverse_media"), reputation.get("screening_result")),
        REPUTATION_ALIASES,
    )
    if reputation_signal is None:
        if _as_bool(reputation.get("unresolved_sanctions_match")):
            reputation_signal = "unresolved_sanctions"
        elif _as_bool(reputation.get("credible_association")):
            reputation_signal = "credible_association"
        elif _as_bool(reputation.get("regulatory_action")):
            reputation_signal = "regulatory"
    reputation_rule = _resolve(REPUTATION_RULES, reputation_signal)

    # -- prior relationship ------------------------------------------------
    relationship_signal = _alias(
        _first(relationship.get("history"), relationship.get("prior_relationship")),
        RELATIONSHIP_ALIASES,
    )
    if relationship_signal is None:
        if _as_bool(relationship.get("previously_suspended"), relationship.get("blacklisted")):
            relationship_signal = "prior_suspended"
        elif _as_bool(relationship.get("prior_issues")):
            relationship_signal = "prior_issues"
        elif _as_bool(relationship.get("existing_vendor")):
            relationship_signal = "prior_qualified"
    relationship_rule = _resolve(RELATIONSHIP_RULES, relationship_signal)

    scored = (
        ("jurisdiction", jurisdiction_signal, jurisdiction_rule),
        ("ownership_control", ownership_signal, ownership_rule),
        ("government_exposure", government_signal, government_rule),
        ("payment_structure", payment_signal, payment_rule),
        ("annual_spend", spend_signal, spend_rule),
        ("nature_sensitivity", nature_signal, nature_rule),
        ("adverse_media", reputation_signal, reputation_rule),
        ("prior_relationship", relationship_signal, relationship_rule),
    )

    return [
        FactorResult(
            factor=factor,
            label=FACTOR_LABELS[factor],
            weight=FACTOR_WEIGHTS[factor],
            score=rule.score,
            rule=rule.rule,
            basis=rule.basis,
            signal=signal,
        )
        for factor, signal, rule in scored
    ]


# --------------------------------------------------------------------------
# Weighted score + tier
# --------------------------------------------------------------------------

def weighted_score(factors: List[FactorResult]) -> Decimal:
    """Weighted sum of factor scores, half-up to two decimals, clamped to the
    policy's 1.00-3.00 range. Half-up rounding is used rather than Python's
    default so a score never lands below a band edge because of banker-style
    rounding (2.205 -> 2.21, not 2.20)."""
    total = Decimal("0")
    for factor in factors:
        total += Decimal(str(factor.weight)) * factor.score
    quantised = total.quantize(QUANTUM, rounding=ROUND_HALF_UP)
    if quantised < MIN_SCORE:
        return MIN_SCORE
    if quantised > MAX_SCORE:
        return MAX_SCORE
    return quantised


def tier_for(score: Decimal) -> Tuple[str, str, str]:
    """Map a weighted score to (tier label, due-diligence level, refresh cycle)."""
    for upper, label, dd_level, cycle in TIER_BANDS:
        if score <= upper:
            return label, dd_level, cycle
    # Unreachable: the last band extends to MAX_SCORE and scores are clamped.
    return TIER_BANDS[-1][1], TIER_BANDS[-1][2], TIER_BANDS[-1][3]


def is_high_tier(tier: str) -> bool:
    return tier == TIER_HIGH


@dataclass
class RiskAssessment:
    """Deterministic assessment result. Contains no LLM-authored text."""

    factors: List[FactorResult]
    weighted_risk_score: Decimal
    assigned_risk_tier: str
    due_diligence_level: str
    refresh_cycle: str
    unknown_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factors": [f.to_dict() for f in self.factors],
            "weighted_risk_score": float(self.weighted_risk_score),
            "assigned_risk_tier": self.assigned_risk_tier,
            "due_diligence_level": self.due_diligence_level,
            "refresh_cycle": self.refresh_cycle,
            "unknown_factors": list(self.unknown_factors),
        }


def assess(signals: Dict[str, Any]) -> RiskAssessment:
    """Score the eight factors, compute the weighted score, and map the tier.

    This is the base-tier assessment only. It does not apply the mandatory
    Always-EDD overrides — see triggers.apply_mandatory_edd, which wraps this
    result. Keeping the two separate means the pre-override score is always
    available for the dossier, even when a trigger forces the tier.
    """
    factors = score_factors(signals)
    score = weighted_score(factors)
    tier, dd_level, cycle = tier_for(score)
    unknown = [f.factor for f in factors if f.signal is None]
    return RiskAssessment(
        factors=factors,
        weighted_risk_score=score,
        assigned_risk_tier=tier,
        due_diligence_level=dd_level,
        refresh_cycle=cycle,
        unknown_factors=unknown,
    )


# --------------------------------------------------------------------------
# Exported for the wizard's optimistic client-side mirror (scoring.ts)
# --------------------------------------------------------------------------

def factor_catalog() -> List[Dict[str, Any]]:
    """Full weight/anchor table, used to document the matrix in the dossier
    and to keep the TypeScript mirror in scoring.ts honest."""
    catalogue: List[Dict[str, Any]] = []
    rule_sets = {
        "jurisdiction": JURISDICTION_RULES,
        "ownership_control": OWNERSHIP_RULES,
        "government_exposure": GOVERNMENT_RULES,
        "payment_structure": PAYMENT_RULES,
        "annual_spend": SPEND_RULES,
        "nature_sensitivity": NATURE_RULES,
        "adverse_media": REPUTATION_RULES,
        "prior_relationship": RELATIONSHIP_RULES,
    }
    for factor, weight in FACTOR_WEIGHTS.items():
        catalogue.append({
            "factor": factor,
            "label": FACTOR_LABELS[factor],
            "weight": weight,
            "anchors": [
                {"signal": key, "score": float(rule.score), "rule": rule.rule}
                for key, rule in sorted(rule_sets[factor].items())
            ],
        })
    return catalogue
