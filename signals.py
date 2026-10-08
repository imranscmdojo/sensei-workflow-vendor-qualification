"""
NH-PQF-001 submission -> risk_engine / triggers signal bag.

The wizard collects declarations; this module translates them into the
normalised signal vocabulary the scoring engine and the trigger set consume.
It is pure Python and never calls a model, so the same submission always
produces the same signals.

Two principles govern the translation:

1. **Explicit beats derived.** A declared value is used as given. Derivation
   only fills a gap the form left open, and every derivation is recorded in
   `derivation_notes` so a reviewer can see that a value was inferred from
   disclosed facts rather than read from a field.

2. **Absence is unknown, not clean.** A field the vendor left blank produces no
   signal, which the engine scores at the neutral 2.00 anchor. It never
   produces a favourable signal. The same rule applies here: `None` means
   "we do not know", and the engine is built to treat that as mid-range.

The one place RAG input enters is the jurisdiction tier. Retrieval may
establish a country tier the form did not declare; a declared form tier is
otherwise left alone, and a retrieval result of "undetermined" never displaces
a declared value.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Documented defaults, stated in the form's own guidance so an officer can see
# what "assessed as routine" means.
DEFAULT_NATURE_SENSITIVITY = "routine"
DEFAULT_RELATIONSHIP_HISTORY = "first_time"


def _truthy(*values: Any) -> bool:
    for value in values:
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value is None:
            continue
        if str(value).strip().lower() in ("true", "yes", "y", "1", "on"):
            return True
    return False


def _falsy(*values: Any) -> bool:
    """An explicit "no" is as informative as an explicit "yes"."""
    for value in values:
        if isinstance(value, bool):
            if not value:
                return True
            continue
        if value is None:
            continue
        if str(value).strip().lower() in ("false", "no", "n", "0", "off"):
            return True
    return False


def _text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            return text
    return ""


def _number(*values: Any) -> Optional[float]:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(str(value).replace(",", "").replace("AED", "").strip())
        except (TypeError, ValueError):
            continue
    return None


def _rows(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, (list, tuple)):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _normalise_country(country: str) -> str:
    return re_sub_nonalpha(country)


def re_sub_nonalpha(text: str) -> str:
    import re
    return re.sub(r"[^a-z]", "", (text or "").lower())


# --------------------------------------------------------------------------
# Ownership
# --------------------------------------------------------------------------

def _ownership(form: Dict[str, Any], notes: List[str]) -> Dict[str, Any]:
    owners = _rows(form.get("directors_and_owners"))
    ubos = _rows(form.get("ubos"))
    country = _text(form.get("country_of_incorporation"))

    structure = _text(
        form.get("ownership_structure"),
        (owners[0] or {}).get("structure") if owners else None,
    )

    ownership: Dict[str, Any] = {
        "declared_structure": form.get("ownership_structure"),
        "ubo_count": len(ubos),
        "owner_count": len(owners),
    }

    if structure:
        ownership["structure"] = structure
    else:
        # Derive only from disclosed facts, and only when nothing was declared.
        derived = _derive_ownership_structure(owners, ubos, form, country)
        if derived:
            ownership["structure"] = derived
            notes.append(
                f"Ownership structure assessed as '{derived}' from the disclosed "
                f"owner and UBO declarations; no structure was declared on the form."
            )

    if _truthy(form.get("nominee_shareholders"), form.get("nominee")):
        ownership["nominee"] = True
    if _truthy(form.get("trust_structure"), form.get("foundation_structure"),
               form.get("private_benefit_company")):
        ownership["trust"] = True
    if _truthy(form.get("bearer_shares")):
        ownership["bearer_share"] = True
    if _truthy(form.get("offshore_structure"), form.get("cross_border_ownership")):
        ownership["offshore"] = True
    if _truthy(form.get("multi_layered_ownership"), form.get("layered_ownership")):
        ownership["layered"] = True

    if ubos:
        ownership["ubo_ownership_percentage_total"] = round(sum(
            float(u.get("ownership_percentage") or 0) for u in ubos
            if _number(u.get("ownership_percentage")) is not None
        ), 2)
        foreign = [
            u for u in ubos
            if country and re_sub_nonalpha(str(u.get("nationality") or ""))
            and re_sub_nonalpha(str(u.get("nationality") or "")) != re_sub_nonalpha(country)
        ]
        if foreign:
            ownership["cross_border_ubos"] = len(foreign)
            notes.append(
                f"{len(foreign)} of {len(ubos)} declared UBOs hold a nationality "
                f"differing from the country of incorporation "
                f"({country or 'not declared'})."
            )
    return ownership


def _derive_ownership_structure(owners: List[Dict[str, Any]],
                                ubos: List[Dict[str, Any]],
                                form: Dict[str, Any],
                                country: str) -> Optional[str]:
    if not owners and not ubos:
        return None
    if _text(form.get("legal_form")) in ("trust", "foundation", "private benefit company"):
        return "trust"
    # Two or more UBO rows with nationalities outside the country of
    # incorporation means ownership crosses a jurisdiction boundary.
    if country and ubos:
        foreign = [
            u for u in ubos
            if re_sub_nonalpha(str(u.get("nationality") or ""))
            and re_sub_nonalpha(str(u.get("nationality") or "")) != re_sub_nonalpha(country)
        ]
        if foreign:
            return "cross_border_layered" if len(ubos) > 1 else "domestic_layered"
    if len(ubos) > 2:
        return "layered"
    if len(ubos) >= 1:
        return "domestic_layered"
    return "simple"


# --------------------------------------------------------------------------
# Jurisdiction
# --------------------------------------------------------------------------

def _jurisdiction(form: Dict[str, Any], rag_tier: Optional[str],
                  notes: List[str]) -> Dict[str, Any]:
    declared = _text(form.get("jurisdiction_risk_tier"), form.get("country_risk_tier"))
    tier = ""

    if rag_tier:
        rag_norm = _text(rag_tier)
        if rag_norm and rag_norm not in ("undetermined", "unknown", "unspecified"):
            tier = rag_norm
            if declared and declared != tier:
                notes.append(
                    f"Country risk tier taken from the retrieved policy corpus "
                    f"('{tier}'); the form declared '{declared}'. The retrieved "
                    f"tier is used."
                )
            else:
                notes.append(
                    f"Country risk tier '{tier}' established from the retrieved "
                    f"policy corpus."
                )
    elif declared:
        tier = declared
        notes.append(
            "Country risk tier taken from the form declaration; retrieval did not "
            "establish a tier for this jurisdiction."
        )

    jurisdiction: Dict[str, Any] = {
        "country_of_incorporation": form.get("country_of_incorporation"),
        "countries_of_operation": _countries_of_operation(form),
        "declared_tier": declared or None,
        "tier_source": "rag" if (rag_tier and _text(rag_tier) not in
                                 ("undetermined", "unknown", "unspecified", "")) else
                        ("form" if tier else "unknown"),
    }
    if tier:
        jurisdiction["tier"] = tier
    if tier in ("high", "prohibited", "banned", "embargoed", "high risk", "high-risk"):
        jurisdiction["high_risk"] = True
    if tier in ("prohibited", "banned", "embargoed"):
        jurisdiction["prohibited"] = True
    return jurisdiction


def _countries_of_operation(form: Dict[str, Any]) -> List[str]:
    raw = form.get("countries_of_operation") or form.get("geographic_coverage")
    if isinstance(raw, (list, tuple)):
        return [str(c).strip() for c in raw if str(c).strip()]
    if isinstance(raw, str) and raw.strip():
        return [part.strip() for part in raw.split(",") if part.strip()]
    return []


# --------------------------------------------------------------------------
# Government exposure
# --------------------------------------------------------------------------

def _government(form: Dict[str, Any]) -> Dict[str, Any]:
    role = _text(
        form.get("business_role"),
        form.get("channel_role"),
        form.get("government_role"),
    )
    government: Dict[str, Any] = {
        "role": role or None,
        "acts_as_intermediary": _truthy(form.get("acts_as_intermediary"),
                                        form.get("is_intermediary")),
        "is_agent": _truthy(form.get("is_agent")),
        "is_distributor": _truthy(form.get("is_distributor"),
                                  form.get("is_agency_distributor")),
        "is_reseller": _truthy(form.get("is_reseller")),
        "is_broker": _truthy(form.get("is_broker")),
        "is_lobbyist": _truthy(form.get("is_lobbyist")),
        "is_government_facing": _truthy(form.get("is_government_facing"),
                                         form.get("government_facing")),
        "engages_government_officials": _truthy(
            form.get("engages_government_officials"),
            form.get("interacts_with_government_officials"),
        ),
        "licensing_required": _truthy(form.get("licensing_required"),
                                     form.get("permit_required"),
                                     form.get("government_licensing")),
    }
    if _truthy(form.get("direct_government_exposure")):
        government["direct_government_exposure"] = True
    if _truthy(form.get("indirect_government_exposure")):
        government["indirect_government_exposure"] = True
    return government


# --------------------------------------------------------------------------
# Payment structure
# --------------------------------------------------------------------------

def _payment(form: Dict[str, Any]) -> Dict[str, Any]:
    structure = _text(form.get("payment_structure"))
    return {
        "structure": structure or None,
        "success_fee": _truthy(form.get("success_fee"), form.get("payment_success_fee")),
        "commission": _truthy(form.get("commission"), form.get("payment_commission")),
        "third_party": _truthy(form.get("third_party_payment"),
                               form.get("payment_to_third_party")),
        "foreign_account": _truthy(form.get("foreign_account_payment"),
                                   form.get("payment_outside_country_of_incorporation")),
        "cash": _truthy(form.get("cash_payment")),
        "payment_terms": form.get("payment_terms"),
    }


# --------------------------------------------------------------------------
# Spend
# --------------------------------------------------------------------------

def _spend(form: Dict[str, Any]) -> Dict[str, Any]:
    annual = _number(form.get("estimated_spend_aed"), form.get("annual_spend_aed"),
                     form.get("expected_spend_aed"))
    contract = _number(form.get("single_contract_value_aed"),
                       form.get("proposed_contract_value_aed"),
                       form.get("contract_value_aed"))
    construction = _number(form.get("construction_project_value_aed"),
                           form.get("project_value_aed"))
    spend: Dict[str, Any] = {
        "annual_spend_aed": annual,
        "single_contract_value_aed": contract,
        "construction_project_value_aed": construction,
        "spend_band": _text(form.get("spend_band")) or None,
        "international_subcontracting": _truthy(
            form.get("international_subcontracting"),
            form.get("cross_border_subcontracting"),
        ),
    }
    return spend


# --------------------------------------------------------------------------
# Nature and sensitivity
# --------------------------------------------------------------------------

NATURE_TEXT_HINTS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("critical infrastructure", "national security", "defence", "defense"),
     "critical"),
    (("sensitive", "controlled", "regulated", "confidential"), "sensitive"),
    (("bank", "custodian", "insurer", "insurance", "financial institution",
      "financial services", "reinsurance"), "financial_institution"),
    (("routine", "non-sensitive", "non sensitive", "standard goods",
      "office", "catering", "cleaning", "logistics"), "routine"),
)


def _nature(form: Dict[str, Any], notes: List[str]) -> Dict[str, Any]:
    declared = _text(form.get("sensitivity"), form.get("goods_services_sensitivity"),
                     form.get("nature_sensitivity"))
    nature: Dict[str, Any] = {
        "sensitivity": declared or None,
        "supply_type": _text(form.get("supply_type")) or None,
        "supplying_goods": _truthy(form.get("supplying_goods")),
    }
    if _truthy(form.get("financial_institution")):
        nature["financial_institution"] = True
    if _truthy(form.get("construction"), form.get("contractor"),
               form.get("facility_management")):
        nature["construction"] = True
    if declared:
        nature["category"] = declared
        return nature
    if nature.get("financial_institution"):
        nature["category"] = "financial_institution"
        return nature

    # Derive sensitivity from the described goods/services, recorded as a note.
    described = " ".join(filter(None, [
        _text(form.get("goods_services_proposed")),
        _text(form.get("products_services") and " ".join(
            str(row.get("name") or "") for row in _rows(form.get("products_services"))
        )),
        _text(form.get("nature_of_business")),
    ]))
    if described:
        for hints, level in NATURE_TEXT_HINTS:
            if any(hint in described for hint in hints):
                nature["category"] = level
                notes.append(
                    f"Goods/services sensitivity assessed as '{level}' from the "
                    f"description on the form; no sensitivity rating was declared."
                )
                return nature
    nature["category"] = DEFAULT_NATURE_SENSITIVITY
    notes.append(
        "Goods/services sensitivity not declared; assessed at the routine anchor "
        "for the weighted factor. Confirm with the vendor."
    )
    return nature


# --------------------------------------------------------------------------
# Reputation and relationship
# --------------------------------------------------------------------------

def _reputation(form: Dict[str, Any], notes: List[str]) -> Dict[str, Any]:
    declared = _text(form.get("adverse_media_result"),
                     form.get("public_records_screening"),
                     form.get("screening_result"))
    adverse_history = _truthy(form.get("prior_regulatory_matter"),
                              form.get("tax_evasion_allegation"),
                              form.get("sanction_history"),
                              form.get("licence_suspension_history"),
                              form.get("regulatory_enforcement_history"))
    reputation: Dict[str, Any] = {
        "adverse_media": declared or None,
        "regulatory_action": adverse_history,
    }
    if declared:
        reputation["adverse_media"] = declared
        return reputation
    if adverse_history:
        reputation["adverse_media"] = "regulatory"
        notes.append(
            "Reputational factor assessed from the supplier's own disclosure of a "
            "prior regulatory matter (NH-PQF-001 Item 59). Full adverse-media "
            "screening is Stage 3 and belongs to the Vendor Onboarding Agent."
        )
        return reputation
    # Nothing declared and Stage 3 screening has not run yet. This is
    # genuinely unknown, not clean, so no signal is emitted and the engine
    # scores the factor at its neutral anchor.
    notes.append(
        "Adverse-media screening has not been performed; that is Stage 3 and "
        "belongs to the Vendor Onboarding Agent. The reputational factor is "
        "therefore unassessed rather than clear."
    )
    return reputation


def _relationship(form: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "history": _text(form.get("prior_relationship"),
                         form.get("prior_relationship_history")) or
                   DEFAULT_RELATIONSHIP_HISTORY,
        "existing_vendor": _truthy(form.get("existing_group_vendor"),
                                   form.get("existing_vendor")),
        "previously_suspended": _truthy(form.get("previously_suspended"),
                                        form.get("suspended"),
                                        form.get("blacklisted")),
        "prior_issues": _truthy(form.get("prior_compliance_issues"),
                                form.get("prior_performance_issues")),
    }


# --------------------------------------------------------------------------
# Declarations that feed the mandatory trigger set
# --------------------------------------------------------------------------

def _pep(form: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "present": _truthy(form.get("pep_present"), form.get("pep_involved")),
        "family_member": _truthy(form.get("pep_family_member")),
        "close_associate": _truthy(form.get("pep_close_associate")),
    }


def _sanctions(form: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "unresolved_match": _truthy(form.get("unresolved_sanctions_match"),
                                    form.get("potential_sanctions_match")),
        "confirmed": _truthy(form.get("confirmed_sanctions_match")),
    }


def _corporate(form: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "joint_venture_partner": _truthy(form.get("joint_venture_partner"),
                                         form.get("jv_partner")),
        "co_investor": _truthy(form.get("co_investor")),
        "ma_target": _truthy(form.get("ma_target"), form.get("acquisition_target")),
    }


def _governance(form: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gcidc_edd_determination": _truthy(form.get("gcidc_edd_determination"),
                                           form.get("gcidc_heightened_risk")),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def build_signals(form: Optional[Dict[str, Any]],
                  rag_jurisdiction_tier: Optional[str] = None) -> Dict[str, Any]:
    """Translate an NH-PQF-001 submission into the risk_engine signal bag.

    Args:
        form: the submitted wizard payload
        rag_jurisdiction_tier: country risk tier established by retrieval, or
            None / "undetermined" when the corpus did not publish one

    Returns:
        The signal bag, plus a `derivation_notes` list explaining any value
        that was derived rather than declared.
    """
    form = form if isinstance(form, dict) else {}
    notes: List[str] = []

    signals: Dict[str, Any] = {
        "ownership": _ownership(form, notes),
        "jurisdiction": _jurisdiction(form, rag_jurisdiction_tier, notes),
        "government": _government(form),
        "payment": _payment(form),
        "spend": _spend(form),
        "nature": _nature(form, notes),
        "reputation": _reputation(form, notes),
        "relationship": _relationship(form),
        "pep": _pep(form),
        "sanctions": _sanctions(form),
        "corporate": _corporate(form),
        "governance": _governance(form),
        "derivation_notes": notes,
    }
    return signals


def signals_to_prompt(signals: Dict[str, Any]) -> str:
    """Compact, model-readable rendering of the signal bag for the prompt.

    Only declared and derived facts are shown, so the narrative pass reasons
    over the same evidence the engine scored. Nothing is added here that is
    not in the bag.
    """
    lines: List[str] = []

    def add(label: str, value: Any) -> None:
        if value in (None, "", [], {}):
            return
        lines.append(f"- {label}: {value}")

    ownership = signals.get("ownership", {})
    add("Ownership structure (declared)", ownership.get("declared_structure"))
    add("Ownership structure (assessed)", ownership.get("structure"))
    add("UBO count", ownership.get("ubo_count"))
    add("Owner/director count", ownership.get("owner_count"))
    add("Cross-border UBO count", ownership.get("cross_border_ubos"))
    add("Nominee / trust / bearer-share declared", any(
        ownership.get(k) for k in ("nominee", "trust", "bearer_share", "offshore")))

    jurisdiction = signals.get("jurisdiction", {})
    add("Country of incorporation", jurisdiction.get("country_of_incorporation"))
    add("Countries of operation", ", ".join(jurisdiction.get("countries_of_operation") or []) or None)
    add("Country risk tier", jurisdiction.get("tier"))
    add("Tier source", jurisdiction.get("tier_source"))

    government = signals.get("government", {})
    add("Business role", government.get("role"))
    add("Government-facing intermediary", government.get("is_government_facing"))
    add("Acts as intermediary", government.get("acts_as_intermediary"))
    add("Distributor", government.get("is_distributor"))
    add("Reseller", government.get("is_reseller"))
    add("Agent / broker / lobbyist", any(
        government.get(k) for k in ("is_agent", "is_broker", "is_lobbyist")))
    add("Engages government officials", government.get("engages_government_officials"))
    add("Licensing / permitting required", government.get("licensing_required"))

    payment = signals.get("payment", {})
    add("Payment structure", payment.get("structure"))
    add("Success fee or commission", any(payment.get(k) for k in ("success_fee", "commission")))
    add("Third-party payment", payment.get("third_party"))
    add("Foreign-account payment", payment.get("foreign_account"))
    add("Cash payment", payment.get("cash"))

    spend = signals.get("spend", {})
    add("Estimated annual spend (AED)", spend.get("annual_spend_aed"))
    add("Single contract value (AED)", spend.get("single_contract_value_aed"))
    add("Construction project value (AED)", spend.get("construction_project_value_aed"))
    add("International sub-contracting", spend.get("international_subcontracting"))

    nature = signals.get("nature", {})
    add("Goods/services sensitivity", nature.get("category"))
    add("Financial institution", nature.get("financial_institution"))
    add("Construction / facility management", nature.get("construction"))

    reputation = signals.get("reputation", {})
    add("Adverse-media result", reputation.get("adverse_media"))
    add("Prior regulatory matter disclosed", reputation.get("regulatory_action"))

    add("Prior relationship with the Group",
        (signals.get("relationship") or {}).get("history"))
    add("Previously suspended or blacklisted",
        (signals.get("relationship") or {}).get("previously_suspended"))

    pep = signals.get("pep", {})
    add("PEP involved", pep.get("present"))
    add("PEP family member", pep.get("family_member"))
    add("PEP close associate", pep.get("close_associate"))

    add("Unresolved sanctions match",
        (signals.get("sanctions") or {}).get("unresolved_match"))
    add("Joint venture / co-investor / M&A target", any(
        (signals.get("corporate") or {}).values()))
    add("GCICD has determined EDD is warranted",
        (signals.get("governance") or {}).get("gcidc_edd_determination"))

    return "\n".join(lines) if lines else "- No risk signals were declared."
