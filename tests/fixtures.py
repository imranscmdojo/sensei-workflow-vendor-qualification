"""
Test fixtures: one fully complete NH-PQF-001 submission, and the deltas used
to take it apart.

`complete_form()` is a submission that passes the Zero-Hallucination gate with
no gaps. It is deliberately a compliant first-time UAE supplier with a
strategic spend, so a single fixture exercises the happy path AND the
mandatory EDD override (STRATEGIC_SPEND, at AED 2,500,000) and the bank
confirmation document gate (over AED 375,000).

`without(form, *fields)` returns a copy with fields removed, for testing what
the completeness gate rejects.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable

# Documents required of every supplier (items 60-81, "All suppliers"), plus
# 63 (VAT, because the supplier is VAT registered) and 73 (bank confirmation,
# because the spend is over AED 375,000). The reference supplier also declares
# ISO 9001 and HSE/ESG certifications and audited financials, so 82 (quality /
# information-security certificates), 83 (ESG compliance) and 84 (audited
# balance sheet) apply here too.
_ALL_DOCUMENTS = [
    "60", "61", "62", "64", "66", "67", "68", "69", "63", "73", "82", "83", "84",
]


def complete_form() -> Dict[str, Any]:
    """A submission that satisfies every blocking rule in validation.FIELD_RULES."""
    return copy.deepcopy({
        # 1. Company details
        "legal_name": "Apex Gulf Technical Solutions LLC",
        "registered_address": "Unit 2401, Marina Gate Tower, Dubai Marina, Dubai, UAE",
        "country_of_incorporation": "United Arab Emirates",
        "date_of_incorporation": "2015-04-12",
        "year_of_commencement": "2015",
        "trade_license_no": "CN-1094821",
        "trade_license_expiry": "2027-04-11",
        "vat_registration_status": "registered",
        "vat_registration_no": "100293847500003",
        "iso_hse_certifications": "ISO 9001:2015, ISO 45001:2018",
        "certifications": ["iso_9001", "hse_esg"],

        # 2. Nature of business
        "nature_of_business": "Engineering services",
        "goods_services_proposed": (
            "Turnkey mechanical and electrical maintenance services across "
            "Group industrial facilities."
        ),
        "geographic_coverage": "United Arab Emirates, Kingdom of Saudi Arabia",

        # 3. Authorised representative
        "authorized_representative_name": "Layla Al-Mansouri",
        "authorized_representative_designation": "Commercial Manager",
        "authorized_representative_id": "784-1990-1234567-1",
        "authorized_representative_id_expiry": "2029-08-30",
        "authorized_representative_authority_basis": "Board resolution dated 2026-01-20",
        "compliance_contact": "compliance@apexgulf.example",

        # 4. Owners and UBOs
        "directors_and_owners": [
            {
                "name": "Rashid Al-Farsi",
                "position": "Managing Director",
                "nationality": "United Arab Emirates",
            },
            {
                "name": "Sara Al-Farsi",
                "position": "Shareholder",
                "nationality": "United Arab Emirates",
            },
        ],
        "ultimate_parent_company": "N/A",
        "ubos": [
            {
                "name": "Rashid Al-Farsi",
                "nationality": "United Arab Emirates",
                "ownership_percentage": 60,
            },
            {
                "name": "Sara Al-Farsi",
                "nationality": "United Arab Emirates",
                "ownership_percentage": 40,
            },
        ],
        "ownership_structure_chart": "upload://ownership-structure-v2.pdf",

        # 5. Client references — three is the policy minimum
        "client_references": [
            {"client_name": "Emirates Industrial Group", "contact_details": "procurement@eig.example"},
            {"client_name": "Gulf Petrochemical Co", "contact_details": "contracts@gpc.example"},
            {"client_name": "Mashreq Facilities", "contact_details": "vendor.ops@mashreqf.example"},
        ],

        # 6. Turnover — three years, ascending
        "turnover_year_1": 18_400_000,
        "turnover_year_2": 15_900_000,
        "turnover_year_3": 12_750_000,
        "number_of_employees": 145,
        "financial_statements_audited": "Yes",

        # 7. Bank and payment
        "bank_name_branch_country": "Emirates NBD, Dubai Marina Branch, UAE",
        "bank_account_name": "Apex Gulf Technical Solutions LLC",
        "bank_account_number": "1234567890123",
        "bank_iban": "AE070331234567890123456",
        "payment_terms": "Net 30 from invoice",
        "estimated_spend_aed": 2_500_000,

        # 8. FTA risk declarations — all answered, all "no"
        "address_change_declaration": "no",
        "management_change_declaration": "no",
        "intermediary_declaration": "no",
        "subcontracting_declaration": "no",
        "regulatory_history_declaration": "no",

        # 9. Products / services
        "products_services": [{"name": "Mechanical maintenance", "code": "MECH-01"}],

        # 10. Documents
        "documents": {code: f"upload://doc-{code}.pdf" for code in _ALL_DOCUMENTS},

        # 12. Declaration and signature
        "supplier_declaration": "yes",
        "authorized_signatory_name": "Rashid Al-Farsi",
        "authorized_signatory_designation": "Managing Director",
        "date_signed": "2026-09-28",

        # Risk-signal declarations consumed by signals.py and triggers.py
        "payment_structure": "standard",
        "sensitivity": "routine",
        "prior_relationship": "first_time",
        "existing_group_vendor": False,
        "pep_present": False,
        "supplying_goods": False,
    })


def without(form: Dict[str, Any], *fields: str) -> Dict[str, Any]:
    """A copy of `form` with `fields` removed."""
    clone = copy.deepcopy(form)
    for field in fields:
        clone.pop(field, None)
    return clone


def with_documents(form: Dict[str, Any], codes: Iterable[str]) -> Dict[str, Any]:
    """A copy of `form` carrying exactly the given document codes."""
    clone = copy.deepcopy(form)
    clone["documents"] = {code: f"upload://doc-{code}.pdf" for code in codes}
    return clone


def without_extra_evidence(form: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of `form` that claims no certifications and no audited accounts.

    The reference supplier declares ISO 9001, HSE/ESG and audited financials, so
    items 82, 83 and 84 are mandatory for it. A test about some *other*
    conditional document must not also trip those three, or it stops being able
    to tell which document caused the block.
    """
    clone = copy.deepcopy(form)
    clone["certifications"] = ["none"]
    clone["financial_statements_audited"] = "No"
    return clone


ALL_SUPPLIER_DOC_CODES = ("60", "61", "62", "64", "66", "67", "68", "69")
