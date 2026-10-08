"""
NH-PQF-001 completeness validator — the Zero-Hallucination gate.

The Tool 1 guardrail is explicit:

    "Never assume missing financial figures, trade licence numbers, or UBO
     details. If mandatory fields in Form NH-PQF-001 are missing, immediately
     set qualification_status to REJECTED_INCOMPLETE and request missing data."

This module is that gate. It is pure Python and runs before any LLM call, so
an incomplete submission is rejected without spending a token or risking a
model inventing a trade licence number.

Field list is transcribed from the actual instrument:
    "NH-PQF-001-Rev.09 - Vendor Prequalification & FTA Verification Form",
    effective 1 October 2026, supersedes Rev.06, 16 sections, items 1-138.

Three classes of requirement, mirroring the form's own
"Mandatory unless marked 'where applicable'" instruction:

    MANDATORY   Always blocks qualification. Items the form marks MANDATORY or
                that apply to all suppliers.
    CONDITIONAL Blocks only when the form's own trigger is present. Examples:
                the VAT certificate blocks only for a VAT-registered supplier;
                the agency agreement blocks only when the supplier declares an
                intermediary, agent, reseller, or broker role.
    ADVISORY    Recorded and surfaced, never blocks. "Provide at least two
                where available" style guidance.

An empty string, a whitespace string, a zero, an empty list, and the literal
string "N/A" are all treated as missing unless the rule sets allow_na. A vendor
that writes "N/A" into a field that has no N/A option has not supplied data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

MIN_CLIENT_REFERENCES = 3
BANK_CONFIRMATION_THRESHOLD_AED = 375_000.0
SMALL_SUPPLY_EXCEPTION_THRESHOLD_AED = 100_000.0

# Item 20 certifications, as a structured multi-select. The wizard sends the
# selected `value` keys in `certifications`; anything outside this set is
# rejected rather than stored, so a downstream condition cannot be triggered by
# a free-text string that merely looks like a certification.
CERTIFICATION_QUALITY = "iso_9001"
CERTIFICATION_INFOSEC = "iso_27001_soc2"
CERTIFICATION_ESG = "hse_esg"
CERTIFICATION_NONE = "none"

CERTIFICATION_VALUES: Tuple[str, ...] = (
    CERTIFICATION_QUALITY,
    CERTIFICATION_INFOSEC,
    CERTIFICATION_ESG,
    CERTIFICATION_NONE,
)

REQUIREMENT_MANDATORY = "MANDATORY"
REQUIREMENT_CONDITIONAL = "CONDITIONAL"
REQUIREMENT_ADVISORY = "ADVISORY"

# Tokens that mean "not supplied". "N/A" is only a valid answer where a rule
# explicitly allows it, because most rows of NH-PQF-001 have no N/A option.
MISSING_TOKENS = {"", "-", "--", "n/a", "na", "n.a.", "none", "nil", "null",
                  "tbc", "tbd", "unknown", "not provided", "not available",
                  "not applicable"}


# --------------------------------------------------------------------------
# Conditional triggers, derived from the form's own guidance text
# --------------------------------------------------------------------------

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


def _selected(form: Dict[str, Any], field_name: str) -> List[str]:
    """Normalise a multi-select answer to a list of known option keys.

    Accepts the list the wizard sends and also a comma-separated string, so a
    hand-built or legacy payload is read rather than silently treated as empty.
    Unknown keys are dropped: a condition must never fire on a value the schema
    does not recognise.
    """
    raw = form.get(field_name)
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in re.split(r"[,\n]", raw)]
    elif isinstance(raw, (list, tuple, set)):
        parts = [str(p).strip().lower() for p in raw]
    else:
        return []
    seen: List[str] = []
    for part in parts:
        if part in CERTIFICATION_VALUES and part not in seen:
            seen.append(part)
    return seen


CONDITIONS: Dict[str, Tuple[str, Callable[[Dict[str, Any]], bool]]] = {
    # Item 18/19: TRN required only for a VAT-registered supplier.
    "vat_registered": (
        "Supplier selected a VAT registration status that requires a TRN",
        lambda f: _text(_get(f, "vat_registration_status")) in
        ("registered", "vat registered", "taxable", "registered for vat",
         "vat applicable"),
    ),
    # Items 57/58/79/80: goods-only controls.
    "goods_supply": (
        "Supplier is supplying goods, so origin and title controls apply",
        lambda f: _truthy(_get(f, "supplying_goods")) or
        _text(_get(f, "supply_type")) in ("goods", "goods and services", "materials"),
    ),
    # Items 55/77: agency, reseller, or intermediary role declared.
    "intermediary_or_reseller": (
        "Supplier declared an intermediary, agent, reseller, or broker role",
        lambda f: _truthy(_get(f, "acts_as_intermediary"), _get(f, "is_distributor"),
                          _get(f, "is_reseller"), _get(f, "is_agent"),
                          _get(f, "is_broker"), _get(f, "is_lobbyist"),
                          _get(f, "is_government_facing")) or
        _text(_get(f, "business_role")) in
        ("intermediary", "agent", "distributor", "reseller", "broker", "lobbyist",
         "government-facing intermediary", "government facing intermediary"),
    ),
    # Items 56/78: sub-contracting declared.
    "subcontracting": (
        "Supplier declared that a third party will materially perform the supply",
        lambda f: _truthy(_get(f, "uses_subcontractor"),
                          _get(f, "third_party_performs_supply"),
                          _get(f, "international_subcontracting")),
    ),
    # Items 49/50/51/76/113: unusual payment structures.
    "unusual_payment": (
        "Supplier declared a third-party, foreign-account, or cash payment",
        lambda f: _truthy(_get(f, "third_party_payment"),
                          _get(f, "foreign_account_payment"),
                          _get(f, "cash_payment"),
                          _get(f, "payment_to_third_party"),
                          _get(f, "payment_outside_country_of_incorporation")),
    ),
    "foreign_account": (
        "Payment into an account outside the country of incorporation declared",
        lambda f: _truthy(_get(f, "foreign_account_payment"),
                          _get(f, "payment_outside_country_of_incorporation")),
    ),
    # Item 59/81: adverse matter disclosed.
    "adverse_matter": (
        "Supplier disclosed a tax-evasion allegation, sanction, licence suspension, or enforcement",
        lambda f: _truthy(_get(f, "prior_regulatory_matter"),
                          _get(f, "tax_evasion_allegation"),
                          _get(f, "sanction_history"),
                          _get(f, "licence_suspension_history"),
                          _get(f, "regulatory_enforcement_history")),
    ),
    # Items 53/74: frequent address changes declared.
    "frequent_address_change": (
        "Registered or operating address changed more than twice in the last 12 months",
        lambda f: _truthy(_get(f, "frequent_address_changes"),
                          _get(f, "address_changed_more_than_twice")),
    ),
    # Items 54/75: frequent management changes declared.
    "frequent_management_change": (
        "Key managers or NH contact persons changed more than twice in the last 12 months",
        lambda f: _truthy(_get(f, "frequent_management_changes"),
                          _get(f, "management_changed_more_than_twice")),
    ),
    # Items 73/111/112: bank confirmation threshold.
    "spend_over_375k": (
        "Supplies over or expected over AED 375,000 in 12 months",
        lambda f: _over_threshold(f, BANK_CONFIRMATION_THRESHOLD_AED),
    ),
    "spend_over_100k": (
        "AED 100,000 cumulative threshold exceeded or expected",
        lambda f: _over_threshold(f, SMALL_SUPPLY_EXCEPTION_THRESHOLD_AED),
    ),
    # Item 20 -> Item 82: a quality or information-security certification was
    # selected, so the certificate itself is required.
    "quality_or_infosec_certified": (
        "Supplier declared an ISO 9001 or ISO 27001 / SOC 2 certification",
        lambda f: bool(
            {CERTIFICATION_QUALITY, CERTIFICATION_INFOSEC} & set(_selected(f, "certifications"))
        ),
    ),
    # Item 20 -> Item 83: HSE / ESG certification declared.
    "esg_certified": (
        "Supplier declared HSE / ESG compliance certification",
        lambda f: CERTIFICATION_ESG in _selected(f, "certifications"),
    ),
    # Item 40a -> Item 84: audited financial statements declared.
    "financial_statements_audited": (
        "Supplier declared that its financial statements are independently audited",
        lambda f: _truthy(_get(f, "financial_statements_audited")),
    ),
}


def _get(form: Dict[str, Any], key: str) -> Any:
    return (form or {}).get(key)


def _over_threshold(form: Dict[str, Any], threshold: float) -> bool:
    """The form's spend triggers are driven by the estimated 12-month value
    (item 91) but the wizard may instead hold only the actual annual spend."""
    for key in ("estimated_spend_aed", "expected_spend_aed", "annual_spend_aed",
                "single_contract_value_aed", "proposed_contract_value_aed"):
        value = _number((form or {}).get(key))
        if value is not None and value >= threshold:
            return True
    declared = _text(_get(form, "spend_band"))
    if declared in ("over_375k", "100k_to_375k", "over_100k", "375k_plus",
                    "100_000_to_375_000", "375_000_plus"):
        band_floor = {
            "over_375k": 375_000.0, "375k_plus": 375_000.0, "375_000_plus": 375_000.0,
            "100k_to_375k": 100_000.0, "100_000_to_375_000": 100_000.0,
            "over_100k": 100_000.0,
        }[declared]
        return band_floor >= threshold
    return False


# --------------------------------------------------------------------------
# Field rules
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldRule:
    field: str
    label: str
    form_item: str
    section: str
    requirement: str = REQUIREMENT_MANDATORY
    kind: str = "text"
    condition: Optional[str] = None
    allow_na: bool = False
    min_items: int = 1
    row_fields: Tuple[str, ...] = ()
    min_length: int = 0
    pattern: Optional[str] = None

    @property
    def blocking(self) -> bool:
        return self.requirement in (REQUIREMENT_MANDATORY, REQUIREMENT_CONDITIONAL)


FIELD_RULES: Tuple[FieldRule, ...] = (
    # -- 1. Company Details ------------------------------------------------
    FieldRule("legal_name", "Company / Legal Name", "Item 5", "1. COMPANY DETAILS"),
    FieldRule("registered_address", "Registered Address", "Item 10", "1. COMPANY DETAILS",
              min_length=5),
    FieldRule("country_of_incorporation", "Country of Incorporation", "Item 8",
              "1. COMPANY DETAILS"),
    FieldRule("date_of_incorporation", "Date of Incorporation", "Item 8",
              "1. COMPANY DETAILS"),
    FieldRule("year_of_commencement", "Year of Commencement of Business", "Item 9",
              "1. COMPANY DETAILS", kind="year"),
    FieldRule("trade_license_no", "Trade Licence Number", "Item 14",
              "1. COMPANY DETAILS", min_length=2),
    FieldRule("trade_license_expiry", "Trade Licence Expiry Date", "Item 15",
              "1. COMPANY DETAILS"),
    FieldRule("vat_registration_status", "VAT Registration Status", "Item 18",
              "1. COMPANY DETAILS"),
    FieldRule("vat_registration_no", "TRN VAT Number", "Item 19",
              "1. COMPANY DETAILS", requirement=REQUIREMENT_CONDITIONAL,
              condition="vat_registered"),
    # Item 20. Was one free-text box ("State N/A if none"); now a structured
    # multi-select so the answer is machine-readable and can drive the
    # certificate upload slots. "Other / None" is an explicit option, so a
    # supplier with no certifications still answers rather than leaving it blank.
    FieldRule("certifications", "ISO / Quality / HSE / InfoSec Certifications",
              "Item 20", "1. COMPANY DETAILS", kind="multi_select"),

    # -- 2. Nature of Business & Product Range -----------------------------
    FieldRule("nature_of_business", "Nature of Business (primary category)", "Item 21",
              "2. NATURE OF BUSINESS & PRODUCT RANGE"),
    FieldRule("goods_services_proposed",
              "Goods / Services Proposed to be Supplied to NH", "Item 23",
              "2. NATURE OF BUSINESS & PRODUCT RANGE", min_length=10),
    FieldRule("geographic_coverage", "Geographic Coverage / Delivery Locations",
              "Item 24", "2. NATURE OF BUSINESS & PRODUCT RANGE",
              requirement=REQUIREMENT_ADVISORY),

    # -- 3. Authorised Representative & Key Contacts -----------------------
    FieldRule("authorized_representative_name", "Authorized Representative Name",
              "Item 25", "3. AUTHORIZED REPRESENTATIVE & KEY CONTACTS"),
    FieldRule("authorized_representative_designation",
              "Authorized Representative Designation / Capacity", "Item 26",
              "3. AUTHORIZED REPRESENTATIVE & KEY CONTACTS"),
    # Item 30 is marked MANDATORY in the form itself.
    FieldRule("authorized_representative_id",
              "Emirates ID / Passport Reference of Authorized Representative", "Item 30",
              "3. AUTHORIZED REPRESENTATIVE & KEY CONTACTS", min_length=3),
    FieldRule("authorized_representative_id_expiry",
              "Emirates ID / Passport Expiry", "Item 30",
              "3. AUTHORIZED REPRESENTATIVE & KEY CONTACTS"),
    FieldRule("authorized_representative_authority_basis",
              "Authority Basis / Validity (POA, board resolution, authorization letter)",
              "Item 29", "3. AUTHORIZED REPRESENTATIVE & KEY CONTACTS"),
    FieldRule("compliance_contact", "Primary Compliance Contact", "Item 34",
              "3. AUTHORIZED REPRESENTATIVE & KEY CONTACTS",
              requirement=REQUIREMENT_ADVISORY),

    # -- 4. Directors, Partners, Shareholders, Owners -----------------------
    # The form asks for the table; the policy requires UBOs. At least one
    # named owner is mandatory, and the table must include a position.
    FieldRule("directors_and_owners", "Directors / Partners / Shareholders / Owners",
              "Item 34 table", "4. DIRECTORS, PARTNERS, SHAREHOLDERS, OWNERS",
              kind="table", min_items=1, row_fields=("name", "position", "nationality")),
    FieldRule("ultimate_parent_company", "Ultimate Parent Company", "Item 35",
              "4. DIRECTORS, PARTNERS, SHAREHOLDERS, OWNERS", allow_na=True),
    # The policy's prequalification criteria require the UBO Declaration
    # (Appendix B); an undeclared UBO set is an incomplete pack.
    FieldRule("ubos", "Ultimate Beneficial Owners (with ownership percentage)",
              "Item 34 / UBO Declaration", "4. DIRECTORS, PARTNERS, SHAREHOLDERS, OWNERS",
              kind="table", min_items=1,
              row_fields=("name", "nationality", "ownership_percentage")),
    FieldRule("ownership_structure_chart", "Ownership Structure Chart", "Item 34",
              "4. DIRECTORS, PARTNERS, SHAREHOLDERS, OWNERS",
              requirement=REQUIREMENT_ADVISORY),

    # -- 5. Major Clients & References -------------------------------------
    FieldRule("client_references", "Major Client References (minimum three required)",
              "Item 35 table", "5. MAJOR CLIENTS & REFERENCES",
              kind="table", min_items=MIN_CLIENT_REFERENCES,
              row_fields=("client_name", "contact_details")),
    FieldRule("public_review_links", "Reliable Public Review Links", "Item 36",
              "5. MAJOR CLIENTS & REFERENCES", requirement=REQUIREMENT_ADVISORY),

    # -- 6. Annual Sales Turnover ------------------------------------------
    FieldRule("turnover_year_1", "Annual Sales Turnover - Year 1 (most recent)",
              "Item 37", "6. ANNUAL SALES TURNOVER", kind="number"),
    FieldRule("turnover_year_2", "Annual Sales Turnover - Year 2", "Item 38",
              "6. ANNUAL SALES TURNOVER", kind="number"),
    FieldRule("turnover_year_3", "Annual Sales Turnover - Year 3", "Item 39",
              "6. ANNUAL SALES TURNOVER", kind="number"),
    FieldRule("number_of_employees", "Number of Employees", "Item 40",
              "6. ANNUAL SALES TURNOVER", kind="number",
              requirement=REQUIREMENT_ADVISORY),
    # Item 40a. A declaration, not an open field: an unanswered audit status is
    # precisely the gap this catches, so only an explicit Yes or No counts.
    # "Yes" also makes the Item 84 audited-accounts document mandatory.
    FieldRule("financial_statements_audited",
              "Are financial statements independently audited?",
              "Item 40a", "6. ANNUAL SALES TURNOVER", kind="declaration"),

    # -- 7. Bank, Payment & Commercial Terms -------------------------------
    FieldRule("bank_name_branch_country", "Bank Name, Branch & Country", "Item 41",
              "7. BANK, PAYMENT & COMMERCIAL TERMS"),
    FieldRule("bank_account_name", "Bank Account Name (must match legal or trade name)",
              "Item 42", "7. BANK, PAYMENT & COMMERCIAL TERMS"),
    FieldRule("bank_account_number", "Bank Account Number", "Item 43",
              "7. BANK, PAYMENT & COMMERCIAL TERMS", min_length=3),
    FieldRule("bank_iban", "IBAN", "Item 44", "7. BANK, PAYMENT & COMMERCIAL TERMS",
              min_length=6),
    FieldRule("swift_code", "SWIFT Code", "Item 45", "7. BANK, PAYMENT & COMMERCIAL TERMS",
              requirement=REQUIREMENT_CONDITIONAL, condition="foreign_account"),
    FieldRule("payment_terms", "Payment Terms", "Item 46",
              "7. BANK, PAYMENT & COMMERCIAL TERMS",
              requirement=REQUIREMENT_ADVISORY),
    FieldRule("unusual_payment_explanation",
              "Explanation for any third-party, foreign-account, or cash payment",
              "Item 52", "7. BANK, PAYMENT & COMMERCIAL TERMS",
              requirement=REQUIREMENT_CONDITIONAL, condition="unusual_payment",
              min_length=10),

    # -- 8. FTA Supplier Risk & Commercial Disclosures ---------------------
    # These are declarations: the answer must be present (yes or no), because
    # an unanswered risk indicator is exactly the case the policy targets.
    FieldRule("address_change_declaration",
              "Address changed more than twice in the previous 12 months?", "Item 53",
              "8. FTA SUPPLIER RISK & COMMERCIAL DISCLOSURES", kind="declaration"),
    FieldRule("management_change_declaration",
              "Key managers or NH contacts changed more than twice in 12 months?",
              "Item 54", "8. FTA SUPPLIER RISK & COMMERCIAL DISCLOSURES",
              kind="declaration"),
    FieldRule("intermediary_declaration",
              "Will the supplier act as intermediary, agent, reseller, or broker?",
              "Item 55", "8. FTA SUPPLIER RISK & COMMERCIAL DISCLOSURES",
              kind="declaration"),
    FieldRule("subcontracting_declaration",
              "Will any subcontractor or third party materially perform the supply?",
              "Item 56", "8. FTA SUPPLIER RISK & COMMERCIAL DISCLOSURES",
              kind="declaration"),
    FieldRule("regulatory_history_declaration",
              "Tax-evasion allegation, sanction, licence suspension, or enforcement in prior 5 years?",
              "Item 59", "8. FTA SUPPLIER RISK & COMMERCIAL DISCLOSURES",
              kind="declaration"),

    # -- 9. Products & Services Listing ------------------------------------
    FieldRule("products_services", "Products / Services Offered", "Item 58 table",
              "9. PRODUCTS & SERVICES LISTING", kind="table", min_items=1,
              row_fields=("name",), requirement=REQUIREMENT_ADVISORY),

    # -- 10. Mandatory & Conditional Documents -----------------------------
    FieldRule("documents", "Mandatory supporting documents", "Items 60-81",
              "10. MANDATORY & CONDITIONAL DOCUMENTS", kind="documents",
              row_fields=("document_code", "attachment_reference")),
    FieldRule("supplier_declaration", "Supplier Declaration (certification of accuracy)",
              "Item 82", "12. SUPPLIER ACKNOWLEDGEMENT & SIGNATURE", kind="declaration"),
    FieldRule("authorized_signatory_name", "Authorized Signatory Name", "Item 83",
              "12. SUPPLIER ACKNOWLEDGEMENT & SIGNATURE"),
    FieldRule("authorized_signatory_designation", "Authorized Signatory Designation",
              "Item 84", "12. SUPPLIER ACKNOWLEDGEMENT & SIGNATURE"),
    FieldRule("date_signed", "Date of Signature", "Item 87",
              "12. SUPPLIER ACKNOWLEDGEMENT & SIGNATURE"),

    # -- Part 2 internal: spend estimate drives the bank-confirmation gate --
    FieldRule("estimated_spend_aed", "Estimated NH spend (previous / next 12 months)",
              "Item 91", "13. PROCUREMENT SUPPLIER ASSESSMENT", kind="number",
              requirement=REQUIREMENT_ADVISORY),
)

# Documents the form marks "All suppliers" — these block outright.
ALL_SUPPLIER_DOCUMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("60", "Code of Conduct acknowledgement (signed)", "Item 60"),
    ("61", "Confidentiality Agreement / NDA (signed)", "Item 61"),
    ("62", "Valid Trade Licence / incorporation certificate", "Item 62"),
    ("64", "Company Profile", "Item 64"),
    ("66", "Power of Attorney / board resolution", "Item 66"),
    ("67", "Copy of Emirates ID or passport of the authorized representative",
     "Item 67"),
    ("68", "Bank details confirmation letter", "Item 68"),
    ("69", "Registered / operating address evidence", "Item 69"),
)

# Documents that block only when their trigger applies.
CONDITIONAL_DOCUMENTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("63", "VAT Registration Certificate / TRN evidence", "Item 63", "vat_registered"),
    ("73", "Written confirmation from an authorized UAE bank", "Item 73", "spend_over_375k"),
    ("74", "Evidence explaining frequent address changes", "Item 74",
     "frequent_address_change"),
    ("75", "Evidence explaining frequent management changes", "Item 75",
     "frequent_management_change"),
    ("76", "Third-party / foreign-account payment agreement", "Item 76",
     "unusual_payment"),
    ("77", "Agency / reseller / intermediary agreement", "Item 77",
     "intermediary_or_reseller"),
    ("78", "Subcontracting agreement / details", "Item 78", "subcontracting"),
    ("79", "Certificate of origin / authenticity", "Item 79", "goods_supply"),
    ("80", "Evidence of ownership / right to sell goods", "Item 80", "goods_supply"),
    ("81", "Regulatory correspondence / remediation evidence", "Item 81",
     "adverse_matter"),
    ("82", "Quality & Information Security Certificates (ISO / SOC 2)", "Item 82",
     "quality_or_infosec_certified"),
    ("83", "ESG / Code of Conduct Compliance Document", "Item 83",
     "esg_certified"),
    ("84", "Audited Balance Sheet / Financial Report (Last 2 Years)", "Item 84",
     "financial_statements_audited"),
)

RULES_BY_FIELD: Dict[str, FieldRule] = {r.field: r for r in FIELD_RULES}

YEAR_RE = re.compile(r"^\d{4}$")
IBAN_RE = re.compile(r"^[A-Z]{2}[0-9A-Z]{10,30}$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Presence checks
# --------------------------------------------------------------------------

def _is_missing(value: Any, allow_na: bool = False) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False  # a declared yes/no is a supplied answer
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    text = str(value).strip()
    if not text:
        return True
    if not allow_na and text.lower() in MISSING_TOKENS:
        return True
    return False


def _is_declaration(value: Any) -> bool:
    """A yes/no disclosure counts as answered when the supplier stated yes or
    no. A blank, a dash, or "N/A" does not: the point of the declaration is
    that the supplier answered it."""
    if isinstance(value, bool):
        return True
    text = _text(value)
    return text in ("yes", "no", "y", "n", "true", "false", "1", "0")


def _row_missing(row: Any, required: Sequence[str]) -> List[str]:
    if not isinstance(row, dict):
        return list(required)
    missing = []
    for key in required:
        if _is_missing(row.get(key)):
            missing.append(key)
    return missing


@dataclass
class MissingField:
    field: str
    label: str
    form_item: str
    section: str
    requirement: str
    reason: str
    condition: Optional[str] = None
    details: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "field": self.field,
            "label": self.label,
            "form_item": self.form_item,
            "section": self.section,
            "requirement": self.requirement,
            "reason": self.reason,
        }
        if self.condition:
            payload["condition"] = self.condition
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class ValidationReport:
    is_complete: bool
    missing: List[MissingField] = field(default_factory=list)
    advisory: List[MissingField] = field(default_factory=list)
    active_conditions: List[str] = field(default_factory=list)

    @property
    def missing_blocking(self) -> List[MissingField]:
        return self.missing

    def missing_field_names(self) -> List[str]:
        return [m.field for m in self.missing]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_complete": self.is_complete,
            # Counts alongside the lists so the wizard can render a progress
            # figure without walking the arrays.
            "missing_count": len(self.missing),
            "advisory_count": len(self.advisory),
            "missing_fields": [m.to_dict() for m in self.missing],
            "advisory_fields": [m.to_dict() for m in self.advisory],
            "active_conditions": list(self.active_conditions),
        }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _active_conditions(form: Dict[str, Any]) -> Dict[str, bool]:
    active: Dict[str, bool] = {}
    for name, (description, predicate) in CONDITIONS.items():
        try:
            active[name] = bool(predicate(form or {}))
        except Exception:  # noqa: BLE001 - a broken condition must not crash intake
            active[name] = False
    return active


def _document_codes(form: Dict[str, Any]) -> Dict[str, str]:
    """Map document code -> attachment reference, tolerating either a list of
    {document_code, attachment_reference} objects or a plain dict."""
    codes: Dict[str, str] = {}
    documents = (form or {}).get("documents")
    if isinstance(documents, dict):
        for code, value in documents.items():
            codes[str(code).strip()] = "" if value is None else str(value).strip()
    elif isinstance(documents, (list, tuple)):
        for entry in documents:
            if not isinstance(entry, dict):
                continue
            code = entry.get("document_code") or entry.get("code") or entry.get("item")
            if code is None:
                continue
            reference = (entry.get("attachment_reference")
                         or entry.get("reference")
                         or entry.get("file_name")
                         or entry.get("file")
                         or "")
            codes[str(code).strip()] = str(reference).strip()
    return codes


def has_document(form: Dict[str, Any], code: str) -> bool:
    """True when the submission carries a non-empty reference for `code`.

    Public counterpart to `_document_codes`, for callers outside the validator
    that need to know whether a specific evidence item was attached — the
    Tool 2 handoff uses it for Item 84.
    """
    return not _is_missing(_document_codes(form or {}).get(str(code)))


def validate(form: Optional[Dict[str, Any]]) -> ValidationReport:
    """Validate an NH-PQF-001 submission for completeness.

    Returns a report whose `missing` list contains only requirement failures
    that block qualification. Advisory gaps are reported separately and never
    affect `is_complete`.
    """
    form = form if isinstance(form, dict) else {}
    active = _active_conditions(form)
    missing: List[MissingField] = []
    advisory: List[MissingField] = []

    for rule in FIELD_RULES:
        if rule.requirement == REQUIREMENT_CONDITIONAL:
            if not rule.condition or not active.get(rule.condition):
                continue

        value = form.get(rule.field)
        problem: Optional[str] = None
        details: Optional[List[str]] = None

        if rule.kind == "table":
            if _is_missing(value):
                if rule.min_items > 1:
                    # An empty list has zero rows. Reporting rule.min_items here
                    # claimed "3 found" for an empty table.
                    problem = (f"at least {rule.min_items} entries are required "
                               "(0 found)")
                else:
                    problem = "no entries supplied"
            else:
                rows = list(value) if isinstance(value, (list, tuple)) else [value]
                incomplete: List[str] = []
                for index, row in enumerate(rows, start=1):
                    gaps = _row_missing(row, rule.row_fields)
                    if gaps:
                        incomplete.append(f"row {index}: missing {', '.join(gaps)}")
                if incomplete:
                    problem = "one or more rows are incomplete"
                    details = incomplete
                elif len(rows) < rule.min_items:
                    # Populated but short. The form makes three client references
                    # a pass/fail prequalification criterion, so two complete rows
                    # must fail just as hard as none at all.
                    problem = (f"at least {rule.min_items} entries are required "
                               f"({len(rows)} found)")
        elif rule.kind == "documents":
            codes = _document_codes(form)
            gaps: List[str] = []
            for code, label, item in ALL_SUPPLIER_DOCUMENTS:
                if _is_missing(codes.get(code)):
                    gaps.append(f"{item} {label}")
            for code, label, item, condition in CONDITIONAL_DOCUMENTS:
                if active.get(condition) and _is_missing(codes.get(code)):
                    gaps.append(f"{item} {label}")
            if gaps:
                problem = "required supporting documents are not attached"
                details = gaps
        elif rule.kind == "declaration":
            if not _is_declaration(value):
                problem = "a yes/no answer is required; the declaration is unanswered"
        elif rule.kind == "multi_select":
            # An empty answer blocks. A non-empty answer with no recognised key
            # also blocks, because otherwise a typo would satisfy the field
            # while silently suppressing every certificate it should have pulled
            # in — a false "complete" on an incomplete pack.
            if _is_missing(value):
                problem = "select at least one option, including “Other / None”"
            elif not _selected(form, rule.field):
                problem = ("select at least one recognised option; received "
                           + ", ".join(sorted(str(v) for v in
                                               (value if isinstance(value, (list, tuple, set))
                                                else [value]))))
        else:
            if _is_missing(value, allow_na=rule.allow_na):
                problem = "required by the form but not supplied"
            elif rule.kind == "year" and not YEAR_RE.match(str(value).strip()):
                problem = "expected a four-digit year"
            elif rule.kind == "number" and _number(value) is None:
                problem = "expected a numeric amount"
            elif rule.min_length and len(str(value).strip()) < rule.min_length:
                problem = f"expected at least {rule.min_length} characters"

        if problem is None:
            continue

        entry = MissingField(
            field=rule.field,
            label=rule.label,
            form_item=rule.form_item,
            section=rule.section,
            requirement=rule.requirement,
            reason=problem,
            condition=rule.condition,
            details=details,
        )
        if rule.blocking:
            missing.append(entry)
        else:
            advisory.append(entry)

    return ValidationReport(
        is_complete=not missing,
        missing=missing,
        advisory=advisory,
        active_conditions=[name for name, on in active.items() if on],
    )


def missing_data_request(report: ValidationReport) -> Dict[str, Any]:
    """The actionable 'please send us the rest' payload returned to the client
    with a REJECTED_INCOMPLETE verdict."""
    return {
        "message": (
            "This submission cannot be qualified yet. The fields below are "
            "required by Form NH-PQF-001 (Rev.09) and were not supplied. No "
            "values have been assumed or inferred for any of them."
        ),
        "form": "NH-PQF-001-Rev.09 - Vendor Prequalification & FTA Verification Form",
        "count": len(report.missing),
        "fields": [m.to_dict() for m in report.missing],
    }
