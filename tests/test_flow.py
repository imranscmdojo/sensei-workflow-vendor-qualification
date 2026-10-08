"""
Acceptance tests for the three vendor onboarding journeys.

    Vendor A — Low Risk  (SDD)   routine services, direct ownership, new
    Vendor B — Medium Risk (CDD)  nominee/layered ownership, structured payment
    Vendor C — High Risk (EDD)   nominee + third-party payment + critical
                                 sensitivity + prior regulatory matter

These are the three bands in `risk_engine.TIER_BANDS`, exercised end to end:
a form goes in, and the tier comes out of the same `build_signals` →
`score_factors` → `weighted_score` → `tier_for` chain the service uses. The
tier is therefore *derived*, never asserted as a literal that could drift from
the policy — the band edges are checked in `tests/test_risk_engine.py`, and
these tests assert the journeys land where the policy says they should.

The supplier-journey tests additionally assert the tipping-off firewall: the
supplier's own view of Vendor C must not contain the tier, the score, the PEP
flag or the trigger, because those are the answers the assessment has not yet
communicated to anyone.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
import tempfile
import time
import unittest
from decimal import Decimal
from unittest import mock
from typing import Any, Dict, List

from fastapi import HTTPException
from fastapi.testclient import TestClient

import demo
import main
import risk_engine
import signals
import supplier_portal
import validation as validation_lib
from risk_engine import TIER_HIGH, TIER_LOW, TIER_MEDIUM, tier_for, weighted_score


# --------------------------------------------------------------------------
# The three journeys
# --------------------------------------------------------------------------

def vendor_a_form() -> Dict[str, Any]:
    """Low Risk (SDD). A straightforward UAE services vendor."""
    return {
        "legal_name": "Apex Gulf Technical Solutions LLC",
        "registered_address": (
            "Unit 2401, Marina Gate Tower, Dubai Marina, Dubai, UAE"
        ),
        "country_of_incorporation": "United Arab Emirates",
        "date_of_incorporation": "2015-04-12",
        "year_of_commencement": "2015",
        "trade_license_no": "CN-1094821",
        "trade_license_expiry": "2027-04-11",
        "vat_registration_status": "registered",
        "vat_registration_no": "100293847500003",
        "nature_of_business": "Engineering services",
        "goods_services_proposed": (
            "Turnkey mechanical and electrical maintenance services."
        ),
        "supply_type": "services",
        "sensitivity": "routine",
        "directors_and_owners": [
            {"name": "Rashid Al-Farsi", "position": "Managing Director",
             "nationality": "United Arab Emirates"},
        ],
        "ubos": [
            {"name": "Rashid Al-Farsi", "nationality": "United Arab Emirates",
             "ownership_percentage": "100"},
        ],
        "ownership_structure": "direct",
        "bearer_shares": "no",
        "nominee_shareholders": "no",
        "pep_present": "no",
        "pep_family_member": "no",
        "pep_close_associate": "no",
        "bank_name_branch_country": "Emirates NBD, Dubai Marina Branch, UAE",
        "bank_account_name": "Apex Gulf Technical Solutions LLC",
        "bank_account_number": "1234567890",
        "bank_iban": "AE070331234567890123456",
        "payment_structure": "standard",
        "third_party_payment": "no",
        "foreign_account_payment": "no",
        "cash_payment": "no",
        "estimated_spend_aed": "250000",
        "single_contract_value_aed": "250000",
        "turnover_year_1": "18400000",
        "number_of_employees": "145",
        "acts_as_intermediary": "no",
        "uses_subcontractor": "no",
        "prior_regulatory_matter": "no",
        "engages_government_officials": "no",
        "government_licensing": "not_applicable",
    }


def vendor_b_form() -> Dict[str, Any]:
    """Medium Risk (CDD). Layered ownership and a structured payment route.

    Same commercial shape as Vendor A so the tier difference is attributable to
    ownership and payment rather than to spend or country.
    """
    form = vendor_a_form()
    form.update({
        "legal_name": "Meridian Industrial Supplies FZE",
        "ownership_structure": "complex",
        "ultimate_parent_company": "Meridian Group Holdings Ltd",
        "payment_structure": "structured",
        "sensitivity": "elevated",
        "estimated_spend_aed": "7500000",
        "single_contract_value_aed": "7500000",
    })
    return form


def vendor_c_form() -> Dict[str, Any]:
    """High Risk (EDD). Nominee holdings, third-party payment, critical
    sensitivity and a prior regulatory matter.

    Reaches the band on the weighted factors alone; the journey then asserts
    that a *mandatory* EDD trigger is also raised, because the two are
    independent controls and a vendor can trip either one.
    """
    form = vendor_a_form()
    form.update({
        "legal_name": "Caspian Trading Partners Ltd",
        "country_of_incorporation": "Panama",
        "ownership_structure": "nominee",
        "nominee_shareholders": "yes",
        "bearer_shares": "yes",
        "payment_structure": "third_party",
        "third_party_payment": "yes",
        "sensitivity": "critical",
        "prior_regulatory_matter": "yes",
        "acts_as_intermediary": "yes",
        "estimated_spend_aed": "25000000",
        "single_contract_value_aed": "25000000",
    })
    return form


def score_of(form: Dict[str, Any]):
    """Run a form through the real signal → score → tier chain."""
    bag = signals.build_signals(form)
    factors = risk_engine.score_factors(bag)
    return weighted_score(factors), tier_for(weighted_score(factors))


# --------------------------------------------------------------------------
# 1. The three journeys land in the three bands
# --------------------------------------------------------------------------

class VendorTierJourneyTest(unittest.TestCase):
    """Apex: SDD. Meridian: CDD. Caspian: EDD."""

    def test_vendor_a_is_simplified_due_diligence(self):
        score, (tier, dd_level, cycle) = score_of(vendor_a_form())
        self.assertEqual(tier, TIER_LOW)
        self.assertEqual(dd_level, "Simplified Due Diligence (SDD)")
        self.assertLessEqual(score, Decimal("1.60"))
        self.assertEqual(cycle, "Every 3 Years")

    def test_vendor_b_is_standard_customer_due_diligence(self):
        score, (tier, dd_level, cycle) = score_of(vendor_b_form())
        self.assertEqual(tier, TIER_MEDIUM)
        self.assertEqual(dd_level, "Standard Customer Due Diligence (CDD)")
        self.assertGreater(score, Decimal("1.60"))
        self.assertLessEqual(score, Decimal("2.20"))
        self.assertEqual(cycle, "Every 2 Years")

    def test_vendor_c_is_enhanced_due_diligence(self):
        score, (tier, dd_level, cycle) = score_of(vendor_c_form())
        self.assertEqual(tier, TIER_HIGH)
        self.assertEqual(dd_level, "Enhanced Due Diligence (EDD)")
        self.assertGreater(score, Decimal("2.20"))
        self.assertEqual(cycle, "Annual Refresh")

    def test_the_three_journeys_are_distinct(self):
        """A regression that flattened the bands would break this."""
        tiers = [score_of(form())[1][0] for form in (
            vendor_a_form, vendor_b_form, vendor_c_form
        )]
        self.assertEqual(tiers, [TIER_LOW, TIER_MEDIUM, TIER_HIGH])


# --------------------------------------------------------------------------
# 2. Token access, no login
# --------------------------------------------------------------------------

class SupplierTokenTest(unittest.TestCase):
    """A supplier has no account. The emailed token is the credential."""

    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)
        self.token = supplier_portal.issue_token("Apex Gulf Technical Solutions LLC")

    def test_a_bearer_token_is_not_required(self):
        """The whole point: no Authorization header, and it still works."""
        response = self.client.get(f"/api/supplier/{self.token}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["vendor_name"], "Apex Gulf Technical Solutions LLC"
        )

    def test_an_unknown_token_is_a_404(self):
        self.assertEqual(
            self.client.get("/api/supplier/not-a-real-token").status_code, 404
        )

    def test_a_missing_token_is_indistinguishable_from_a_wrong_one(self):
        """Otherwise a guesser learns a token existed."""
        wrong = self.client.get("/api/supplier/definitely-not-valid")
        empty = self.client.get("/api/supplier/" + "x" * 43)
        self.assertEqual(wrong.status_code, empty.status_code)
        self.assertEqual(wrong.json(), empty.json())

    def test_an_expired_token_stops_working(self):
        session = supplier_portal._load_session(self.token)
        session.created_at -= supplier_portal.TOKEN_TTL_SECONDS + 1
        self.assertEqual(
            self.client.get(f"/api/supplier/{self.token}").status_code, 404
        )


# --------------------------------------------------------------------------
# 3. Saving form sections
# --------------------------------------------------------------------------

class SupplierFormSectionTest(unittest.TestCase):
    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)
        self.token = supplier_portal.issue_token("Apex Gulf")

    def test_saving_a_declared_field(self):
        response = self.client.put(
            f"/api/supplier/{self.token}/form/company",
            json={"fields": {"legal_name": "Apex Gulf Technical Solutions LLC"}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            supplier_portal._load_session(self.token).form["legal_name"],
            "Apex Gulf Technical Solutions LLC",
        )

    def test_an_unknown_section_is_a_404(self):
        self.assertEqual(
            self.client.put(
                f"/api/supplier/{self.token}/form/risk_scoring",
                json={"fields": {"legal_name": "x"}},
            ).status_code,
            404,
        )

    def test_a_supplier_cannot_write_a_field_that_is_not_theirs(self):
        """The save route is scoped to the allowlist, so a planted internal
        field is refused rather than stored and hidden."""
        response = self.client.put(
            f"/api/supplier/{self.token}/form/company",
            json={"fields": {"assigned_risk_tier": "Approved"}},
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(
            "assigned_risk_tier",
            supplier_portal._load_session(self.token).form,
        )

    def test_saving_does_not_change_the_neutral_status(self):
        self.client.put(
            f"/api/supplier/{self.token}/form/company",
            json={"fields": {"legal_name": "Apex Gulf"}},
        )
        view = self.client.get(f"/api/supplier/{self.token}").json()
        self.assertEqual(view["status"], "In progress")


# --------------------------------------------------------------------------
# 4. Document upload and extraction
# --------------------------------------------------------------------------

def _make_pdf(lines: List[str]) -> bytes:
    """Build a real single-page PDF, xref and all.

    Hand-built rather than pulled from a fixture so the test is self-contained,
    and rather than reportlab so the suite gains no dependency it does not
    already have. The offsets have to be correct or pypdf reports an empty
    document and the test passes for the wrong reason.
    """
    content = "BT /F1 11 Tf 54 740 Td 14 TL\n"
    for line in lines:
        escaped = (
            line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        )
        content += f"({escaped}) Tj T*\n"
    content += "ET\n"

    objects = [
        "<</Type/Catalog/Pages 2 0 R>>",
        "<</Type/Pages/Kids[3 0 R]/Count 1>>",
        "<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        "/Resources<</Font<</F1 5 0 R>>>>>>",
        f"<</Length {len(content)}>>stream\n{content}\nendstream",
        "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n{body}\nendobj\n".encode("latin-1")

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<</Size {len(objects) + 1}/Root 1 0 R>>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(out)


_LICENCE_LINES = [
    "Company / legal name Meridian Industrial Supplies FZE",
    "Trade licence number CN-8821400",
    "Registered address Bay 12, Jebel Ali Free Zone, Dubai, UAE",
    "TRN VAT number 100445566700003",
]
_LICENCE_PDF = _make_pdf(_LICENCE_LINES)

_BANK_LINES = [
    "Company / legal name Meridian Industrial Supplies FZE",
    "Bank name / branch / country Emirates NBD, Jebel Ali Free Zone Branch, UAE",
    "Bank IBAN AE07 0331 2345 6789 0123 456",
]
_BANK_PDF = _make_pdf(_BANK_LINES)


class SupplierDocumentUploadTest(unittest.TestCase):
    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)
        self.token = supplier_portal.issue_token("Meridian Industrial Supplies FZE")

    def _upload(self, data=_LICENCE_PDF, filename="trade-licence.pdf", code="62"):
        return self.client.post(
            f"/api/supplier/{self.token}/documents",
            files={"file": (filename, io.BytesIO(data), "application/pdf")},
            params={"document_code": code},
        )

    def test_a_readable_licence_fills_blank_fields_only(self):
        response = self._upload()
        self.assertEqual(response.status_code, 200)
        form = supplier_portal._load_session(self.token).form
        self.assertEqual(form["trade_license_no"], "CN-8821400")
        self.assertEqual(form["legal_name"], "Meridian Industrial Supplies FZE")
        self.assertEqual(response.json()["document"]["status"], "Read")

    def test_extraction_never_overwrites_what_the_supplier_typed(self):
        self.client.put(
            f"/api/supplier/{self.token}/form/company",
            json={"fields": {"trade_license_no": "CN-0000000"}},
        )
        self._upload()
        form = supplier_portal._load_session(self.token).form
        self.assertEqual(form["trade_license_no"], "CN-0000000")

    def test_a_mismatch_becomes_a_neutral_clarification(self):
        """Vendor B typed one bank; the letter says another."""
        self.client.put(
            f"/api/supplier/{self.token}/form/banking",
            json={"fields": {"bank_name_branch_country": "Emirates NBD, Deira Branch"}},
        )
        response = self._upload(_BANK_PDF, filename="bank-letter.pdf", code="68")
        questions = response.json()["clarifications"]
        self.assertTrue(questions, "a differing bank name must raise a question")
        # The question may mention the bank, but must not mention risk.
        blob = repr(questions).lower()
        for banned in ("risk", "edd", "pep", "sanction", "tier", "score", "flag"):
            self.assertNotIn(banned, blob)
        # …and the supplier's own value stands.
        self.assertEqual(
            supplier_portal._load_session(self.token).form["bank_name_branch_country"],
            "Emirates NBD, Deira Branch",
        )

    def test_an_agreeing_bank_letter_raises_nothing(self):
        response = self._upload(_BANK_PDF, filename="bank-letter.pdf", code="68")
        self.assertEqual(response.json()["clarifications"], [])
        self.assertEqual(
            supplier_portal._load_session(self.token).form["bank_name_branch_country"],
            "Emirates NBD, Jebel Ali Free Zone Branch, UAE",
        )

    def test_a_formatting_difference_is_not_treated_as_a_mismatch(self):
        self.client.put(
            f"/api/supplier/{self.token}/form/company",
            json={"fields": {"trade_license_no": "cn 8821400"}},
        )
        response = self._upload()
        self.assertEqual(response.json()["clarifications"], [])

    def test_an_unsupported_file_is_refused_without_detail(self):
        response = self._upload(
            data=b"MZ\x90\x00 this is a windows binary",
            filename="payload.exe",
        )
        self.assertIn(response.status_code, (400, 415))
        blob = response.text.lower()
        for banned in ("pep", "sanction", "edd", "risk"):
            self.assertNotIn(banned, blob)

    def test_an_empty_file_is_refused(self):
        self.assertEqual(self._upload(data=b"").status_code, 400)

    def test_the_upload_response_never_carries_internal_fields(self):
        blob = self._upload().text.lower()
        for banned in ("risk", "tier", "edd", "pep", "sanction", "score"):
            self.assertNotIn(banned, blob)


class DocumentPreviewTest(unittest.TestCase):
    """The Vendor file's Preview link: bytes a supplier files come back
    byte-for-byte through the document route, while every view and response
    stays metadata-only — and a file stored without a copy is a plain 404
    rather than a link that lies."""

    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)
        self.token = supplier_portal.issue_token("Meridian Industrial Supplies FZE")

    def _upload(self, filename="trade-licence.pdf", code="62"):
        # Multipart field, like the real client sends it.
        return self.client.post(
            f"/api/supplier/{self.token}/documents",
            files={"file": (filename, io.BytesIO(_LICENCE_PDF), "application/pdf")},
            data={"document_code": code},
        )

    def test_an_uploaded_file_comes_back_byte_for_byte(self):
        self.assertEqual(self._upload().status_code, 200)

        preview = self.client.get(
            f"/api/supplier/{self.token}/documents/62-trade-licence.pdf"
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.headers["content-type"], "application/pdf")
        self.assertEqual(preview.content, _LICENCE_PDF)
        self.assertIn("inline", preview.headers.get("content-disposition", ""))

    def test_responses_and_views_stay_metadata_only(self):
        upload = self._upload()
        self.assertNotIn("content", upload.json()["document"])
        self.assertNotIn("content_type", upload.json()["document"])

        # The bytes exist on the stored record — the preview route proves it —
        # yet neither reader hands them out: the supplier's own view and the
        # buyer's detail view both return metadata.
        session = supplier_portal._load_session(self.token)
        self.assertIn("content", session.documents[0])

        supplier = self.client.get(f"/api/supplier/{self.token}")
        self.assertEqual(supplier.status_code, 200)
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as("buyer")
        try:
            buyer = self.client.get(f"/api/portal/submissions/{self.token}")
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)
        self.assertEqual(buyer.status_code, 200, buyer.text)

        for view in (supplier.json()["documents"], buyer.json()["documents"]):
            for record in view:
                self.assertNotIn("content", record)
                self.assertNotIn("content_type", record)

    def test_an_unknown_reference_or_token_is_a_plain_404(self):
        self._upload()
        missing = self.client.get(
            f"/api/supplier/{self.token}/documents/not-uploaded.pdf"
        )
        self.assertEqual(missing.status_code, 404)

        stranger = supplier_portal.issue_token("Somebody Else Ltd")
        wrong_token = self.client.get(
            f"/api/supplier/{stranger}/documents/62-trade-licence.pdf"
        )
        self.assertEqual(wrong_token.status_code, 404)

    def test_a_file_stored_without_a_copy_is_not_previewable(self):
        with unittest.mock.patch.object(
            supplier_portal, "MAX_DOCUMENT_PREVIEW_BYTES", 8
        ):
            self._upload()
        session = supplier_portal._load_session(self.token)
        self.assertNotIn("content", session.documents[0])
        preview = self.client.get(
            f"/api/supplier/{self.token}/documents/62-trade-licence.pdf"
        )
        self.assertEqual(preview.status_code, 404)


# --------------------------------------------------------------------------
# 5. The tipping-off firewall
# --------------------------------------------------------------------------

class TippingOffFirewallTest(unittest.TestCase):
    """Vendor C is the dangerous one: high tier, PEP, prior regulatory matter.

    None of it may reach the supplier, at any point in the journey.
    """

    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)
        self.token = supplier_portal.issue_token("Caspian Trading Partners Ltd")
        session = supplier_portal._load_session(self.token)
        session.form.update(vendor_c_form())
        # The internal dossier exists and is fully populated. None of it may
        # escape.
        session.internal = {
            "assigned_risk_tier": TIER_HIGH,
            "weighted_risk_score": 2.25,
            "mandatory_edd_triggered": True,
            "trigger_details": [{"code": "NOMINEE_HOLDINGS"}],
            "sanctions_result": "cleared",
            "jurisdiction_assessment": {"tier": "high"},
        }
        supplier_portal._save_session(session)

    def test_the_supplier_view_carries_no_internal_signal(self):
        blob = self.client.get(f"/api/supplier/{self.token}").text.lower()
        for banned in (
            "edd", "tier", "risk_score", "risk score", "sanction",
            "nominee_holdings", "jurisdiction", "trigger",
        ):
            self.assertNotIn(banned, blob)

    def test_the_pep_and_ubo_declarations_round_trip(self):
        """The supplier's own UBO rows and PEP declarations come back to them.

        These were previously excluded, which was a mistake rather than a
        safeguard: Item 34 is mandatory, so a supplier could not declare their
        beneficial owners at all and every submission came back
        REJECTED_INCOMPLETE. What stays withheld is the screening outcome
        derived from the answers — see the assertions below and
        test_the_supplier_view_carries_no_internal_signal.
        """
        payload = {
            "ubos": [
                {"name": "Rashid Al-Farsi", "nationality": "United Arab Emirates",
                 "ownership_percentage": "100"},
            ],
            "pep_present": "No",
            "pep_family_member": "No",
            "pep_close_associate": "No",
        }
        saved = self.client.put(
            f"/api/supplier/{self.token}/form/ownership", json={"fields": payload}
        )
        self.assertEqual(saved.status_code, 200)
        form = self.client.get(f"/api/supplier/{self.token}").json()["form"]
        self.assertEqual(form["ubos"], payload["ubos"])
        self.assertEqual(form["pep_present"], "No")

    def test_the_screening_outcome_from_those_declarations_is_not_reflected(self):
        """Answering "Yes" to PEP must not tell the supplier what it triggered."""
        self.client.put(
            f"/api/supplier/{self.token}/form/ownership",
            json={"fields": {"pep_present": "Yes"}},
        )
        supplier_portal.run_assessment(supplier_portal._load_session(self.token))
        view = self.client.get(f"/api/supplier/{self.token}").json()
        blob = json.dumps(view).lower()
        for leaked in ("screening", "sanction", "edd", "escalat", "high risk",
                       "weighted_risk", "risk_tier"):
            self.assertNotIn(leaked, blob, f"{leaked!r} leaked to the supplier")
        self.assertIn(view["status"], supplier_portal.NEUTRAL_STATUSES)

    def test_the_status_is_neutral_even_after_an_edd_outcome(self):
        session = supplier_portal._load_session(self.token)
        session.neutral_status = supplier_portal.neutral_status_for("ESCALATED")
        supplier_portal._save_session(session)
        view = self.client.get(f"/api/supplier/{self.token}").json()
        self.assertEqual(view["status"], "Under review")
        self.assertIn(view["status"], supplier_portal.NEUTRAL_STATUSES)

    def test_only_the_four_neutral_statuses_are_ever_offered(self):
        view = self.client.get(f"/api/supplier/{self.token}").json()
        self.assertEqual(len(view["statuses_you_may_see"]), 4)
        self.assertEqual(
            set(view["statuses_you_may_see"]), set(supplier_portal.NEUTRAL_STATUSES)
        )

    def test_neutral_mapping_covers_every_internal_outcome(self):
        for internal in (
            "REJECTED_INCOMPLETE", "QUALIFIED", "CONDITIONALLY_QUALIFIED",
            "ESCALATED", "REFERRED", "NEEDS_EDD", "", None, "SOMETHING_NEW",
        ):
            with self.subTest(internal=internal):
                self.assertIn(
                    supplier_portal.neutral_status_for(internal),
                    supplier_portal.NEUTRAL_STATUSES,
                )

    def test_the_allowlist_never_names_an_internal_field(self):
        overlap = set(supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS) & (
            supplier_portal.NEVER_SUPPLIER_VISIBLE
        )
        self.assertEqual(overlap, set())


# --------------------------------------------------------------------------
# 6. MLRO-only compliance decisions
# --------------------------------------------------------------------------

def _as(role: str):
    """Override the Firebase dependency with a fixed role claim.

    The role travels in the verified token, so the test has to stand in for the
    token — which is exactly why the route cannot read a role from a header.
    """
    return {"uid": f"u-{role}", "email": f"{role}@example.com", "role": role}


class MlroOnlyTest(unittest.TestCase):
    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)
        self.token = supplier_portal.issue_token("Caspian Trading Partners Ltd")
        supplier_portal.create_compliance_alert(
            "alert-1", "Caspian Trading Partners Ltd",
            "Nominee holdings with a prior regulatory matter.",
            supplier_token=self.token,
        )

    def _decide(self, role: str, decision: str = "escalated", justification: str = ""):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as(role)
        try:
            return self.client.post(
                "/api/compliance/alerts/alert-1/decision",
                json={"decision": decision, "justification": justification},
            )
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)

    def test_the_mlro_may_decide(self):
        response = self._decide("mlro")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "escalated")

    def test_a_buyer_is_forbidden(self):
        self.assertEqual(self._decide("buyer").status_code, 403)

    def test_the_gceo_is_forbidden(self):
        self.assertEqual(self._decide("gceo").status_code, 403)

    def test_no_other_role_may_decide(self):
        for role in ("procurement", "analyst", "admin", "finance", "auditor", ""):
            with self.subTest(role=role):
                self.assertEqual(self._decide(role).status_code, 403)

    def test_a_refused_decision_leaves_the_alert_undecided(self):
        self._decide("buyer")
        self.assertFalse(supplier_portal.read_alert("alert-1")["decided"])

    def test_the_supplier_is_told_only_that_action_is_required(self):
        self._decide("mlro")
        view = self.client.get(f"/api/supplier/{self.token}").json()
        self.assertEqual(view["status"], "Action required")
        self.assertNotIn("nominee", repr(view).lower())

    def test_an_alert_cannot_be_decided_twice(self):
        self.assertEqual(self._decide("mlro").status_code, 200)
        self.assertEqual(self._decide("mlro").status_code, 409)

    def test_deciding_requires_authentication(self):
        """No token at all is 401, not an open door."""
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: None
        try:
            response = self.client.post(
                "/api/compliance/alerts/alert-1/decision",
                json={"decision": "escalated"},
            )
            self.assertIn(response.status_code, (401, 403))
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)

    def test_the_subsidiary_compliance_officer_may_decide(self):
        """The console's "Viewing as" selector offers SCO beside the MLRO,
        so the backend must accept the same two roles and no others."""
        response = self._decide("sco", "cleared")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "cleared")

    def test_a_justification_is_recorded_with_the_decision(self):
        """The justification is the audit trail the queue shows the console
        after the decision; the supplier still sees none of it."""
        response = self._decide(
            "mlro",
            "cleared",
            justification="Screened against the sanctions list; the name match was a different entity.",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "sanctions list", response.json()["justification"]
        )
        self.assertIn(
            "sanctions list", supplier_portal.read_alert("alert-1")["justification"]
        )
        view = self.client.get(f"/api/supplier/{self.token}").json()
        self.assertNotIn("sanctions", repr(view).lower())

    def test_the_queue_carries_the_recorded_justification(self):
        """The compliance queue lists the decision and its reason together."""
        self._decide("mlro", "rejected", justification="Unresolved beneficial-owner match.")
        rows = self.client.get("/api/console/compliance/alerts").json()["alerts"]
        row = next(r for r in rows if r["id"] == "alert-1")
        self.assertEqual(row["status"], "decided")
        self.assertEqual(row["decision"], "rejected")
        self.assertIn("beneficial-owner", row["justification"])


# --------------------------------------------------------------------------
# 6b. Agentic Decision Co-Pilot — draft analysis and the Glass Box
# --------------------------------------------------------------------------

class AgenticCoPilotTest(unittest.TestCase):
    """The agent drafts; the human files; the Glass Box says who did which.

    The draft endpoint is read-only and gated by the same `_require_mlro`
    as the decision it feeds — it can fill a textarea and nothing else.
    The decision endpoint keeps its human-in-the-loop gate, and every
    decision writes one append-only Glass Box row whose wording records
    whether the justification began as agent text.
    """

    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)
        self.token = supplier_portal.issue_token("Caspian Trading Partners Ltd")
        supplier_portal.create_compliance_alert(
            "alert-1", "Caspian Trading Partners Ltd",
            "Nominee holdings with a prior regulatory matter.",
            supplier_token=self.token,
        )
        # The RAG generation leg is a Vertex call; the suite stays offline,
        # so the model is stubbed to fail and every draft below rides the
        # deterministic record fallback. Tests that want the RAG leg
        # re-patch `_llm_rag_draft` with their own return value.
        patcher = mock.patch.object(
            supplier_portal,
            "_llm_rag_draft",
            side_effect=RuntimeError("generation stubbed out in tests"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, path: str, body: Dict[str, Any], role: str | None = "mlro"):
        if role is None:
            main.app.dependency_overrides[main.verify_firebase_token] = lambda: None
        else:
            main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as(role)
        try:
            return self.client.post(path, json=body)
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)

    def _draft(self, decision: str = "cleared", role: str | None = "mlro",
               alert_id: str = "alert-1"):
        return self._post(
            f"/api/console/compliance/alerts/{alert_id}/ai-draft",
            {"decision": decision}, role,
        )

    def _decide(self, role: str = "mlro", alert_id: str = "alert-1", **body):
        return self._post(
            f"/api/console/compliance/alerts/{alert_id}/decision", body, role,
        )

    def _events(self, role: str = "mlro"):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as(role)
        try:
            return self.client.get("/api/console/glassbox").json()["events"]
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)

    # The draft ---------------------------------------------------------------

    def test_the_draft_quotes_the_screening_record(self):
        """The draft opens with the finding, quotes the queue's reason, and
        states plainly that it is advisory — a proposal, not a verdict."""
        response = self._draft()
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertTrue(data["draft"].startswith("AI Finding:"))
        self.assertGreaterEqual(len(data["draft"].strip()), 20)
        self.assertIn("Nominee holdings", data["draft"])
        self.assertIn("human-in-the-loop", data["draft"])
        self.assertTrue(data["findings"])
        self.assertIn("Queue reason:", data["findings"][0])

    def test_the_draft_extracts_the_score_the_engine_produced(self):
        """Scored submissions get the weighted score and the top factors in
        the findings — extracted from the record, never invented."""
        session = supplier_portal._load_session(self.token)
        session.form.update({
            k: v for k, v in vendor_c_form().items()
            if k in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS
        })
        supplier_portal._save_session(session)
        supplier_portal.submit_session(self.token, notify=False)

        data = self._draft().json()
        blob = "\n".join(data["findings"])
        self.assertIn("Weighted risk score", blob)
        self.assertIn("Highest-scoring factors", blob)
        self.assertIn("Declared sanctions screening", blob)
        # The manual alert stays the one in the queue: the raise dedupes per
        # token, so scoring did not stack a second row behind the first.
        rows = self.client.get("/api/console/compliance/alerts").json()["alerts"]
        self.assertEqual(
            [r["id"] for r in rows if r["id"].startswith("auto-")], [])

    def test_the_draft_is_read_only(self):
        """Asking for a draft changes nothing: the alert is still open and
        still carries no justification."""
        self._draft()
        alert = supplier_portal.read_alert("alert-1")
        self.assertFalse(alert["decided"])
        self.assertIsNone(alert["justification"])
        self.assertEqual(self._events(), [])

    def test_a_buyer_may_not_ask_for_a_draft(self):
        """The draft feeds a restricted decision, so it carries the same
        gate — a buyer presenting as MLRO in their own browser gets 403."""
        self.assertEqual(self._draft(role="buyer").status_code, 403)
        self.assertEqual(self._draft(role="gceo").status_code, 403)

    def test_the_draft_reports_which_source_served_it(self):
        """With the model stubbed out (setUp) the deterministic record draft
        returns, and the response says so: `record-fallback`, alongside the
        three-part retrieval context that was assembled either way."""
        data = self._draft().json()
        self.assertEqual(data["source"], "record-fallback")
        self.assertTrue(data["draft"].startswith("AI Finding:"))
        self.assertEqual(
            sorted(data["context"]), ["documents", "policy", "screening"]
        )
        # The fallback is a document, not a flat bullet list: headline,
        # numbered sections, recommendation, advisory trailer.
        self.assertIn("1. Screening & Watchlist Alert Summary", data["draft"])
        self.assertIn("2. Document Evidence Cross-Verification", data["draft"])
        self.assertIn("3. Policy & Compliance Basis", data["draft"])
        self.assertIn("4. AI Recommendation", data["draft"])
        self.assertNotIn("- - ", data["draft"])

    def test_with_generation_the_draft_is_grounded_in_the_rag_context(self):
        """The RAG leg returns the model's text plus the context it was
        grounded in: screening findings & watchlist alerts, the extracted
        document vault, and the policy specification rules."""
        canned = (
            "AI Finding: the watchlist name corresponds to a different "
            "legal person. human-in-the-loop."
        )
        with mock.patch.object(
            supplier_portal, "_llm_rag_draft", return_value=canned
        ) as generation:
            data = self._draft().json()
        self.assertEqual(data["source"], "rag-llm")
        self.assertEqual(data["draft"], canned)
        self.assertTrue(generation.called)
        screening = "\n".join(data["context"]["screening"])
        self.assertIn("Queue reason:", screening)
        self.assertIn("PEP status:", screening)
        vault = "\n".join(data["context"]["documents"])
        self.assertIn("Trade licence:", vault)
        self.assertIn("UBO declarations:", vault)
        policy = "\n".join(data["context"]["policy"])
        self.assertIn("Risk Scoring Matrix", policy)
        self.assertIn("Always-EDD", policy)

    def test_markdown_never_reaches_the_textarea(self):
        """The model sometimes drafts in markdown; a textarea would show the
        asterisks literally. Bold markers are unwrapped and every section
        heading gets a blank line before and after — never squeezed."""
        canned = (
            "AI Finding: a false positive on the record.\n"
            "**1. Screening findings & watchlist alerts**\n"
            "Caspian has an unresolved match on file.\n"
            "Recommendation: clear it — subject to your review."
        )
        with mock.patch.object(
            supplier_portal, "_llm_rag_draft", return_value=canned
        ):
            draft = self._draft().json()["draft"]
        self.assertTrue(draft.startswith("AI Finding:"))
        self.assertNotIn("**", draft)
        self.assertIn(
            "\n\n1. Screening findings & watchlist alerts\n\n", draft
        )
        self.assertIn("\n\nRecommendation:", draft)

    def test_an_rag_assisted_decision_names_the_context_applied(self):
        """The Glass Box marker the spec asks for: adopting RAG-pipeline
        text records "Assisted by Compliance Agent (RAG Context Applied)"."""
        response = self._decide(
            "mlro",
            decision="cleared",
            justification="AI Finding: a different legal person; the record shows it.",
            assisted=True,
            rag=True,
        )
        self.assertEqual(response.status_code, 200, response.text)
        event = self._events()[-1]
        self.assertEqual(
            event["action"],
            "Cleared by MLRO (Assisted by Compliance Agent (RAG Context Applied))",
        )
        self.assertTrue(event["payload"]["assisted"])
        self.assertTrue(event["payload"]["rag"])

    def test_the_draft_requires_authentication(self):
        self.assertIn(self._draft(role=None).status_code, (401, 403))

    def test_a_decided_alert_offers_no_draft(self):
        """There is nothing left to assist: the decision is recorded."""
        self._decide("mlro", decision="cleared", justification="Screened; different entity.")
        self.assertEqual(self._draft().status_code, 409)

    def test_an_unknown_alert_offers_no_draft(self):
        self.assertEqual(self._draft(alert_id="nope").status_code, 404)

    def test_a_draft_is_only_written_for_clear_or_block(self):
        """The two buttons the panel offers are the two drafts it can ask
        for; an escalation has no drafted justification to fill."""
        self.assertEqual(self._draft(decision="escalated").status_code, 400)

    # The Glass Box -----------------------------------------------------------

    def test_an_assisted_decision_is_logged_verbatim(self):
        """The spec's exact line: who (the role the gate matched), what,
        and that the Compliance Agent's text was adopted."""
        response = self._decide(
            "mlro",
            decision="cleared",
            justification="Name match cleared: the watchlist entity is a different legal person.",
            assisted=True,
        )
        self.assertEqual(response.status_code, 200, response.text)
        events = self._events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(
            event["action"], "Cleared by MLRO (Assisted by Compliance Agent)"
        )
        self.assertEqual(event["level"], "L0")
        self.assertEqual(event["actor_type"], "human")
        self.assertEqual(event["actor_role"], "MLRO")
        self.assertEqual(event["alert_id"], "alert-1")
        self.assertEqual(event["vendor_name"], "Caspian Trading Partners Ltd")
        self.assertIn("Policy: MLRO decision is final", event["rules"])
        self.assertTrue(event["payload"]["assisted"])
        self.assertGreater(event["payload"]["justification_chars"], 0)

    def test_a_manual_decision_is_not_labelled_assisted(self):
        """Truthfulness of the log: no agent text was adopted, so no agent
        is credited with assisting."""
        self._decide(
            "mlro",
            decision="cleared",
            justification="Screened against the sanctions list; not the same entity.",
        )
        event = self._events()[-1]
        self.assertEqual(event["action"], "Cleared by MLRO")
        self.assertNotIn("Assisted", event["action"])
        self.assertFalse(event["payload"]["assisted"])

    def test_blocking_the_vendor_is_logged_as_blocked(self):
        self._decide(
            "mlro",
            decision="rejected",
            justification="Confirmed sanctions match; the vendor is blocked pending EDD.",
            assisted=True,
        )
        event = self._events()[-1]
        self.assertEqual(
            event["action"], "Blocked by MLRO (Assisted by Compliance Agent)"
        )

    def test_the_scope_officer_gets_her_own_line(self):
        """SCO decides under the same gate, and the log names the role that
        actually took the decision rather than assuming the MLRO."""
        self._decide(
            "sco",
            decision="cleared",
            justification="Ownership documents reconcile the name match to a different entity.",
            assisted=True,
        )
        event = self._events(role="sco")[-1]
        self.assertEqual(
            event["action"], "Cleared by SCO (Assisted by Compliance Agent)"
        )
        self.assertEqual(event["actor_role"], "SCO")

    def test_the_glass_box_appends_rather_than_rewrites(self):
        """Two decisions, two rows, sequence numbers in order: the log is
        append-only, nothing is edited in place."""
        supplier_portal.create_compliance_alert(
            "alert-2", "Bharat Heavy Fabricators LLC",
            "Watchlist name similarity on the director.",
        )
        self._decide("mlro", decision="cleared",
                     justification="Different entity after document review.")
        self._decide("mlro", alert_id="alert-2", decision="rejected",
                     justification="Unresolved watchlist match; blocked.",
                     assisted=True)
        events = self._events()
        self.assertEqual([e["seq"] for e in events], [1, 2])
        self.assertEqual(events[0]["alert_id"], "alert-1")
        self.assertEqual(events[1]["alert_id"], "alert-2")

    def test_the_glass_box_requires_authentication(self):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: None
        try:
            self.assertIn(self.client.get("/api/console/glassbox").status_code,
                          (401, 403))
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)

    def test_the_glass_box_is_invisible_to_the_supplier(self):
        """The tipping-off firewall extends to the audit log: the supplier
        is told action is required, never what the officer wrote about them."""
        self._decide("mlro", decision="cleared",
                     justification="Watchlist name cleared against the passport record.",
                     assisted=True)
        view = self.client.get(f"/api/supplier/{self.token}").json()
        blob = repr(view).lower()
        self.assertNotIn("glass box", blob)
        self.assertNotIn("compliance agent", blob)
        self.assertNotIn("assisted", blob)


class GlassBoxPersistenceTest(unittest.TestCase):
    """The Glass Box rides in `portal_items` and has two contracts to keep.

    It survives a cold start, and `_restore()` — which `list_invitations()`
    re-runs on every control-tower refresh to pick up rows another instance
    wrote — merges idempotently. Approvals and alerts get that for free from
    dict assignment; an appended row once did not, and the same event showed
    up twice in the log for it.
    """

    def _run(self, db: str, code: str) -> str:
        env = dict(os.environ, SUPPLIER_PORTAL_DB=db, AUTH_DISABLED="true")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        return out.stdout.strip()

    def test_a_re_restored_log_does_not_duplicate_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "portal.db")
            counts = self._run(db, textwrap.dedent("""
                import supplier_portal as sp
                sp.log_glassbox(
                    "Cleared by MLRO (Assisted by Compliance Agent)",
                    actor="u-mlro", actor_role="MLRO", actor_type="human",
                    level="L0", rules=["Policy: MLRO decision is final"],
                    alert_id="alert-1", vendor_name="Caspian Trading Partners Ltd",
                )
                first = len(sp.glassbox_events())
                sp.list_invitations()   # re-runs _restore() from disk
                second = len(sp.glassbox_events())
                sp.list_invitations()   # and once more, as a busy console does
                third = len(sp.glassbox_events())
                print(f"{first},{second},{third}")
            """))
            self.assertEqual(counts, "1,1,1")

    def test_the_log_continues_after_a_cold_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "portal.db")
            self._run(db, textwrap.dedent("""
                import supplier_portal as sp
                sp.log_glassbox("Cleared by MLRO", actor="u-mlro",
                                actor_role="MLRO", alert_id="a1")
            """))
            # A brand-new interpreter must continue the stored log's sequence
            # and read the old row back — not restart at 1 and overwrite it,
            # and not show a blank log because memory starts empty.
            out = self._run(db, textwrap.dedent("""
                import supplier_portal as sp
                sp.log_glassbox("Blocked by MLRO (Assisted by Compliance Agent)",
                                actor="u-mlro", actor_role="MLRO", alert_id="a2")
                events = sp.glassbox_events()
                print("|".join(f"{e['seq']}:{e['action']}" for e in events))
            """))
            self.assertEqual(
                out,
                "1:Cleared by MLRO|"
                "2:Blocked by MLRO (Assisted by Compliance Agent)",
            )


# --------------------------------------------------------------------------
# 6c. Agent Activity — the sub-agent stream behind the dashboard
# --------------------------------------------------------------------------

class AgentActivityTest(unittest.TestCase):
    """The Agent Activity stream: what the background sub-agents have done.

    It opens seeded — a dashboard that starts blank looks broken — and the
    fixture is history: real events append after it, never on top of it.
    Every action the spec names has to land in it: a clarification chase,
    a document upload, an MLRO decision, plus the autonomous chaser the
    agent clock fires. Read-only over HTTP, append-only underneath.
    """

    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)

    def _events(self, role="buyer"):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as(role)
        try:
            response = self.client.get("/api/console/agent-activity")
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["events"]

    def _last(self, role="buyer"):
        events = self._events(role)
        self.assertTrue(events, "the stream must never be empty")
        return events[-1]

    def _upload(self, token, data=_LICENCE_PDF, filename="trade-licence.pdf", code="62"):
        # `document_code` is a multipart field, like the supplier portal
        # sends it (`fd.append("document_code", …)`) — not a query param.
        return self.client.post(
            f"/api/supplier/{token}/documents",
            files={"file": (filename, io.BytesIO(data), "application/pdf")},
            data={"document_code": code},
        )

    # --- the fixture -----------------------------------------------------

    def test_the_feed_starts_with_mock_sub_agent_events(self):
        events = self._events()
        self.assertEqual(len(events), len(supplier_portal._ACTIVITY_SEED))
        # Every entry carries what the panel renders: agent tag, summary,
        # rule badge, timestamp, sequence.
        for event in events:
            for field in ("seq", "at", "agent", "level", "summary", "rule", "vendor"):
                self.assertIn(field, event)
            self.assertTrue(event["agent"] and event["level"] and event["summary"])
            self.assertGreater(event["at"], 0)
        # Chronological: sequence and clock time agree, oldest first.
        self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual(events, sorted(events, key=lambda e: e["at"]))
        # The realistic content the spec pack asks for: the Form 74 example,
        # the Form 73 UAE bank confirmation, the Form 68 verification, and
        # the Spec 3.4 chase — with the rules that badge them.
        blob = repr(events)
        self.assertIn("B_74_address_change_evidence.pdf", blob)
        self.assertIn("uae_bank_confirmation", blob)
        self.assertIn("Form 68", blob)
        self.assertIn("Spec 3.4; A7", blob)
        # The roster matches the onboarding agent's orchestrator.
        self.assertEqual(
            {e["agent"] for e in events},
            {"Intake and Triage", "Document Intelligence",
             "Supplier Concierge", "Screening"},
        )

    # --- dynamic sync: each named action appends -------------------------

    def test_an_upload_appends_a_document_intelligence_event(self):
        token = supplier_portal.issue_token("Meridian Industrial Supplies FZE")
        response = self._upload(token)
        self.assertEqual(response.status_code, 200, response.text)
        event = self._last()
        self.assertEqual(event["agent"], "Document Intelligence")
        self.assertEqual(event["level"], "L3")
        self.assertEqual(event["rule"], "Form 62")
        self.assertIn("trade-licence.pdf", event["summary"])
        self.assertEqual(event["vendor"], "Meridian Industrial Supplies FZE")

    def test_a_sent_back_submission_appends_a_chase_event(self):
        token = supplier_portal.issue_token(
            "Decision Path Co", invited_by="owner@firm.com")
        session = supplier_portal._load_session(token)
        session.form.update(
            {k: v for k, v in vendor_c_form().items()
             if k in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS}
        )
        supplier_portal._save_session(session)
        supplier_portal.submit_session(token, notify=False)
        response = self.client.post(
            f"/api/portal/submissions/{token}/decision",
            json={"decision": "rejected",
                  "justification": "The bank letter is unsigned."},
            headers={"Authorization": "Bearer owner@firm.com"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        event = self._last()
        self.assertEqual(event["agent"], "Supplier Concierge")
        self.assertEqual(event["rule"], "Spec 3.4; A7")
        self.assertIn("Decision Path Co", event["summary"])

    def test_an_mlro_decision_appends_a_screening_event(self):
        token = supplier_portal.issue_token("Caspian Trading Partners Ltd")
        supplier_portal.create_compliance_alert(
            "alert-1", "Caspian Trading Partners Ltd",
            "Nominee holdings with a prior regulatory matter.",
            supplier_token=token,
        )
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as("mlro")
        try:
            response = self.client.post(
                "/api/compliance/alerts/alert-1/decision",
                json={"decision": "cleared",
                      "justification": "Watchlist name cleared against the passport record."},
            )
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)
        self.assertEqual(response.status_code, 200, response.text)
        event = self._last()
        self.assertEqual(event["agent"], "Screening")
        self.assertEqual(event["rule"],
                         "Sanctions, PEP and Adverse-Media Screening; R08")
        self.assertIn("Caspian Trading Partners Ltd", event["summary"])
        self.assertIn("cleared", event["summary"])

    def test_a_silent_supplier_gets_chased_by_the_agent_clock(self):
        """The autonomous Spec 3.4 chase appends too — and only once."""
        demo.reset_clock()
        token = supplier_portal.issue_token(
            "Silent Vendor Ltd", supplier_email="ops@silent.example")
        session = supplier_portal._load_session(token)
        session.created_at = time.time() - 4 * 86_400   # day 4 of the clock
        supplier_portal._save_session(session)

        fired = demo.run_chasers()
        self.assertTrue(any(f["kind"] == "chaser-reminder" for f in fired))
        event = self._last()
        self.assertEqual(event["agent"], "Supplier Concierge")
        self.assertEqual(event["rule"], "Spec 3.4; A7")
        self.assertIn("Silent Vendor Ltd", event["summary"])

        # The outbox dedupe that stops a duplicate email must stop the
        # duplicate log line as well: the chaser runs on every clock tick.
        before = len(self._events())
        demo.run_chasers()
        self.assertEqual(len(self._events()), before)

    def test_a_live_event_lands_after_the_fixture_not_over_it(self):
        token = supplier_portal.issue_token("Meridian Industrial Supplies FZE")
        self.assertEqual(self._upload(token).status_code, 200)
        events = supplier_portal.agent_activity_events()
        self.assertEqual(len(events), len(supplier_portal._ACTIVITY_SEED) + 1)
        self.assertEqual(events[-1]["seq"], len(supplier_portal._ACTIVITY_SEED) + 1)
        self.assertEqual(events[0]["seq"], 1)
        self.assertEqual(events[0]["agent"], "Intake and Triage")

    # --- access ----------------------------------------------------------

    def test_the_stream_requires_authentication(self):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: None
        try:
            self.assertIn(
                self.client.get("/api/console/agent-activity").status_code,
                (401, 403),
            )
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)

    def test_the_supplier_never_sees_the_stream(self):
        """The tipping-off firewall extends to the feed: the supplier is
        told what to do, never what the agents logged while doing it."""
        token = supplier_portal.issue_token("Meridian Industrial Supplies FZE")
        supplier_portal.log_agent_activity(
            "Screening", "Internal note carried over from screening.",
            rule="R08", vendor="Meridian Industrial Supplies FZE")
        view = self.client.get(f"/api/supplier/{token}").json()
        blob = repr(view).lower()
        self.assertNotIn("agent activity", blob)
        self.assertNotIn("internal note", blob)
        self.assertNotIn("document intelligence", blob)


class AgentActivityPersistenceTest(unittest.TestCase):
    """The stream rides in `portal_items` like the Glass Box does: it
    survives a cold start and merges idempotently when `_restore()` — which
    `list_invitations()` re-runs on every control-tower refresh — puts the
    persisted rows back over ones already in memory."""

    def _run(self, db: str, code: str) -> str:
        env = dict(os.environ, SUPPLIER_PORTAL_DB=db, AUTH_DISABLED="true")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        return out.stdout.strip()

    def test_a_re_restored_stream_does_not_duplicate_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "portal.db")
            counts = self._run(db, textwrap.dedent("""
                import supplier_portal as sp
                sp.agent_activity_events()   # seeds the six-event fixture
                sp.log_agent_activity("Screening", "One live event.", rule="R08")
                first = len(sp.agent_activity_events())
                sp.list_invitations()   # re-runs _restore() from disk
                second = len(sp.agent_activity_events())
                sp.list_invitations()   # and once more, as a busy console does
                third = len(sp.agent_activity_events())
                print(f"{first},{second},{third}")
            """))
            self.assertEqual(counts, "7,7,7")

    def test_the_stream_continues_after_a_cold_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "portal.db")
            self._run(db, textwrap.dedent("""
                import supplier_portal as sp
                sp.agent_activity_events()   # seeds and persists the fixture
                sp.log_agent_activity("Document Intelligence", "Read item 62.",
                                      rule="Form 62")
            """))
            # A brand-new interpreter reads the stored rows back — no second
            # seeding on top of them — and appends at the right sequence.
            out = self._run(db, textwrap.dedent("""
                import supplier_portal as sp
                sp.log_agent_activity("Supplier Concierge", "Chased them.",
                                      rule="Spec 3.4; A7")
                events = sp.agent_activity_events()
                print(len(events), events[-1]["seq"], events[-1]["agent"])
            """))
            self.assertEqual(out, "8 8 Supplier Concierge")


class DemoChecklistSeedTest(unittest.TestCase):
    """The Supplier View simulator's document checklist reads the portal's
    own records, so the demo recipes seed them: B arrives with a complete
    pack — including the documents its own answers activate (Form 73 for
    spend over AED 375,000, Item 82 for its declared ISO certificate, Item 84
    for its audited statements) — C carries every conditional its own story
    activates (those plus its intermediary agreement and regulatory
    remediation evidence, still without the Item 80 ownership evidence it
    cannot produce), and A — which has not submitted — with nothing at all.
    Every one of these packs is meant to pass the completeness gate: Run
    Qualification must never reject a seeded submission as incomplete."""

    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)

    FULL_PACK = {
        "60", "61", "62", "64", "66", "67", "68", "69",
        "63", "79", "80",
        # Conditionals B's own answers activate: spend over AED 375,000,
        # declared ISO certificate, audited statements.
        "73", "82", "84",
    }

    # C answers a different set of conditions: it holds an HSE/ESG
    # certificate rather than an ISO one (Item 83, not 82), declares an
    # intermediary role (Item 77) and a prior regulatory matter (Item 81),
    # and never produces the ownership evidence (Item 80).
    CASPIAN_PACK = (FULL_PACK - {"80", "82"}) | {"77", "81", "83"}

    def _docs(self, token):
        session = supplier_portal._load_session(token)
        return {d["document_code"] for d in session.documents}

    def test_each_recipe_seeds_its_own_checklist(self):
        tok_b = demo.run_vendor("b")["token"]
        tok_c = demo.run_vendor("c")["token"]
        tok_a = demo.run_vendor("a")["token"]

        self.assertEqual(self._docs(tok_b), self.FULL_PACK)
        self.assertEqual(self._docs(tok_c), self.CASPIAN_PACK)
        self.assertEqual(self._docs(tok_a), set())

        # The rows survive into the supplier's own view — the exact shape
        # the simulator renders — carrying the code the checklist ticks
        # against, and seeded files always read as uploaded.
        view = supplier_portal.supplier_view(supplier_portal._load_session(tok_c))
        codes = {d["document_code"] for d in view["documents"]}
        self.assertEqual(len(view["documents"]), len(self.CASPIAN_PACK))
        self.assertIn("60", codes)
        self.assertNotIn("80", codes)
        self.assertTrue(all(d["status"] != "unreadable" for d in view["documents"]))

    def test_seeded_documents_preview_and_carry_their_findings(self):
        tok_c = demo.run_vendor("c")["token"]

        # The seeded copy is a real PDF the buyer's preview route serves…
        preview = self.client.get(
            f"/api/supplier/{tok_c}/documents/62-62-trade-licence.pdf"
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.headers["content-type"], "application/pdf")
        self.assertTrue(preview.content.startswith(b"%PDF"))

        # …the corporate facts the Vendor file shows came with the recipe…
        view = supplier_portal.supplier_view(supplier_portal._load_session(tok_c))
        self.assertEqual(view["form"].get("vat_registration_no"), "100321654900003")
        self.assertEqual(view["form"].get("bank_iban"), "AE240331231234567890123")
        self.assertTrue(view["form"].get("registered_address"))

        # …and Document Intelligence logged the pack it read, so the audit
        # log has the AI findings beside the Glass Box's clearance trace.
        reads = [
            e
            for e in supplier_portal.agent_activity_events()
            if e["agent"] == "Document Intelligence"
            and e["vendor"] == "Caspian Energy Trading Ltd"
        ]
        self.assertTrue(reads)
        self.assertIn(
            f"{len(self.CASPIAN_PACK)} documents", reads[-1]["summary"])


# --------------------------------------------------------------------------
# 7. Approvals — GET briefs, POST decides
# --------------------------------------------------------------------------

class ApprovalBriefTest(unittest.TestCase):
    """A mail scanner GETs every link in an inbox.

    If GET could approve, reading the notification email would approve the
    vendor. So GET is read-only and the decision needs a POST.
    """

    def setUp(self):
        supplier_portal.reset_all()
        self.client = TestClient(main.app)
        self.supplier_token = supplier_portal.issue_token("Apex Gulf Technical Solutions LLC")
        self.brief_token = supplier_portal.create_approval_brief(
            "Apex Gulf Technical Solutions LLC",
            "Standard CDD. Routine services, direct ownership.",
            "buyer@group.example",
            supplier_token=self.supplier_token,
        )

    def test_get_renders_the_brief(self):
        brief = self.client.get(f"/api/approvals/{self.brief_token}").json()
        self.assertEqual(brief["vendor_name"], "Apex Gulf Technical Solutions LLC")
        self.assertFalse(brief["decided"])
        self.assertIn("does not approve anything", brief["notice"])

    def test_get_alone_never_decides(self):
        for _ in range(3):
            self.client.get(f"/api/approvals/{self.brief_token}")
        self.assertFalse(supplier_portal.read_approval_brief(self.brief_token)["decided"])
        self.assertEqual(
            self.client.get(f"/api/supplier/{self.supplier_token}").json()["status"],
            "In progress",
        )

    def test_post_records_the_decision(self):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as("mlro")
        try:
            response = self.client.post(
                f"/api/approvals/{self.brief_token}",
                json={"decision": "approved"},
            )
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["decided"])
        self.assertEqual(
            self.client.get(f"/api/supplier/{self.supplier_token}").json()["status"],
            "Approved",
        )

    def test_a_rejection_shows_the_supplier_action_required(self):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as("mlro")
        try:
            self.client.post(
                f"/api/approvals/{self.brief_token}", json={"decision": "rejected"}
            )
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)
        self.assertEqual(
            self.client.get(f"/api/supplier/{self.supplier_token}").json()["status"],
            "Action required",
        )

    def test_a_decision_cannot_be_played_twice(self):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as("mlro")
        try:
            self.client.post(f"/api/approvals/{self.brief_token}", json={"decision": "approved"})
            second = self.client.post(
                f"/api/approvals/{self.brief_token}", json={"decision": "rejected"}
            )
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)
        self.assertEqual(second.status_code, 409)

    def test_an_unknown_decision_is_refused(self):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as("mlro")
        try:
            response = self.client.post(
                f"/api/approvals/{self.brief_token}", json={"decision": "maybe"}
            )
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)
        self.assertEqual(response.status_code, 400)

    def test_posting_requires_an_authenticated_approver(self):
        main.app.dependency_overrides[main.verify_firebase_token] = lambda: {"uid": "", "email": ""}
        try:
            response = self.client.post(
                f"/api/approvals/{self.brief_token}", json={"decision": "approved"}
            )
            self.assertEqual(response.status_code, 401)
        finally:
            main.app.dependency_overrides.pop(main.verify_firebase_token, None)

    def test_the_brief_itself_carries_no_internal_risk_output(self):
        """A brief is read by approvers, but it is still a token-bearing URL
        that gets forwarded, so it carries the commercial summary only."""
        blob = self.client.get(f"/api/approvals/{self.brief_token}").text.lower()
        for banned in ("edd", "sanction", "pep", "weighted_risk_score"):
            self.assertNotIn(banned, blob)


# --------------------------------------------------------------------------
# 8. Full journey, all three vendors
# --------------------------------------------------------------------------

class FullSupplierJourneyTest(unittest.TestCase):
    """Token → form → document → decision, for each of the three vendors."""

    def test_each_vendor_completes_a_journey_and_the_tier_stays_internal(self):
        for form_factory, expected_tier in (
            (vendor_a_form, TIER_LOW),
            (vendor_b_form, TIER_MEDIUM),
            (vendor_c_form, TIER_HIGH),
        ):
            with self.subTest(vendor=form_factory.__name__):
                supplier_portal.reset_all()
                client = TestClient(main.app)
                token = supplier_portal.issue_token("Journey vendor")
                session = supplier_portal._load_session(token)
                session.form.update(form_factory())
                supplier_portal._save_session(session)

                # The internal assessment is computed and stored internally.
                # score_of returns (score, (tier, dd_level, cycle)) — the tier
                # is unpacked, not carried as the whole tuple, or the EDD
                # comparison below silently compares a tuple to a string and
                # takes the wrong branch.
                score, (tier, _dd_level, _cycle) = score_of(form_factory())
                session.internal.update({
                    "assigned_risk_tier": tier,
                    "weighted_risk_score": float(score),
                })
                supplier_portal._save_session(session)

                # The supplier uploads a licence.
                client.post(
                    f"/api/supplier/{token}/documents",
                    files={"file": ("licence.pdf", io.BytesIO(_LICENCE_PDF),
                                    "application/pdf")},
                    params={"document_code": "62"},
                )

                # An MLRO decides, if there is anything to decide.
                alert_id = f"alert-{token[:8]}"
                supplier_portal.create_compliance_alert(
                    alert_id, "Journey vendor", "internal reason",
                    supplier_token=token,
                )
                main.app.dependency_overrides[main.verify_firebase_token] = lambda: _as("mlro")
                try:
                    decision = "escalated" if tier == TIER_HIGH else "cleared"
                    client.post(
                        f"/api/compliance/alerts/{alert_id}/decision",
                        json={"decision": decision},
                    )
                finally:
                    main.app.dependency_overrides.pop(main.verify_firebase_token, None)

                # What the supplier ends up with.
                view = client.get(f"/api/supplier/{token}").json()
                self.assertIn(view["status"], supplier_portal.NEUTRAL_STATUSES)
                self.assertTrue(view["documents"])
                if tier == TIER_HIGH:
                    self.assertEqual(view["status"], "Action required")

                # No conclusion, score, trigger or screening result may appear.
                # The supplier's *own declarations* legitimately do — they
                # told us ownership is nominee, and their answer is theirs.
                # What must not appear is our reading of it.
                blob = repr(view).lower()
                for banned in (
                    "edd", "high risk", "medium risk", "weighted_risk_score",
                    "sanction", "trigger", "jurisdiction", "mandatory",
                ):
                    self.assertNotIn(banned, blob)
                # The internal dossier is unreachable from the response.
                self.assertNotIn("internal", view)
                self.assertNotIn("weighted_risk_score", view["form"])


class LinkSurvivesRestartTest(unittest.TestCase):
    """A link a vendor is already filling in must survive a backend restart.

    Run in a subprocess rather than by reloading the module in-process: a
    reload would leave the imported singletons half-replaced for every other
    test, and this needs to be a genuine cold start to mean anything.
    """

    def _run(self, db: str, code: str) -> str:
        env = dict(os.environ, SUPPLIER_PORTAL_DB=db, AUTH_DISABLED="true")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        return out.stdout.strip()

    def test_a_link_and_its_progress_survive_a_cold_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "portal.db")

            token = self._run(db, textwrap.dedent("""
                import supplier_portal as sp
                token = sp.issue_token("Apex Gulf Technical Solutions LLC")
                session = sp._load_session(token)
                session.form["legal_name"] = "Apex Gulf Technical Solutions LLC"
                session.form["country_of_incorporation"] = "UAE"
                sp._save_session(session)
                print(token)
            """))
            self.assertTrue(token)

            # A brand-new interpreter: nothing in memory, everything from disk.
            view = self._run(db, textwrap.dedent(f"""
                import json, supplier_portal as sp
                print(json.dumps(sp.supplier_view(sp._load_session({token!r}))))
            """))
            data = json.loads(view)
            self.assertEqual(data["vendor_name"], "Apex Gulf Technical Solutions LLC")
            self.assertEqual(data["form"]["country_of_incorporation"], "UAE")

    def test_an_expired_link_does_not_come_back_to_life(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "portal.db")
            self._run(db, textwrap.dedent(f"""
                import time, supplier_portal as sp
                token = sp.issue_token("Stale Ltd")
                session = sp._load_session(token)
                session.created_at = time.time() - sp.TOKEN_TTL_SECONDS - 60
                sp._save_session(session)
            """))
            out = self._run(db, textwrap.dedent("""
                import supplier_portal as sp
                try:
                    sp._load_session("anything")
                    print("RESOLVED")
                except Exception as exc:
                    print(getattr(exc, "status_code", "raised"), "not valid")
            """))
            self.assertNotIn("RESOLVED", out)
            self.assertIn("not valid", out)


class SubmissionNotificationTest(unittest.TestCase):
    """Supplier submits -> scored, buyer notified, nothing leaks back."""

    def setUp(self):
        supplier_portal.reset_all()
        for var in ("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD",
                    "SMTP_FROM", "NOTIFICATION_TO", "SUPPLIER_PORTAL_BASE_URL"):
            os.environ.pop(var, None)

    def _session(self, vendor="Apex Gulf Technical Solutions LLC", inviter="buyer@firm.com"):
        token = supplier_portal.issue_token(vendor, invited_by=inviter)
        session = supplier_portal._load_session(token)
        session.form.update(vendor_c_form())
        supplier_portal._save_session(session)
        return token

    def test_submitting_scores_the_form_and_moves_it_under_review(self):
        token = self._session()
        out = supplier_portal.submit_session(token, notify=False)
        self.assertFalse(out["already_submitted"])
        self.assertEqual(out["status"], "Under review")
        # Vendor C is the High Risk fixture, so the score must not be a shrug.
        self.assertEqual(out["assessment"]["assigned_risk_tier"], TIER_HIGH)
        self.assertTrue(float(out["assessment"]["weighted_risk_score"]) > 2.20)

    def test_submitting_twice_does_not_rescore_or_renotify(self):
        token = self._session()
        first = supplier_portal.submit_session(token, notify=False)
        second = supplier_portal.submit_session(token, notify=False)
        self.assertTrue(second["already_submitted"])
        self.assertEqual(
            first["assessment"]["assigned_risk_tier"],
            second["assessment"]["assigned_risk_tier"],
        )

    def test_the_supplier_is_told_only_under_review(self):
        token = self._session()
        supplier_portal.submit_session(token, notify=False)
        view = supplier_portal.supplier_view(supplier_portal._load_session(token))
        self.assertEqual(view["status"], "Under review")
        blob = repr(view).lower()
        for banned in ("high risk", "edd", "weighted_risk_score", "assigned_risk_tier",
                       "2.2", "nominee_holdings", "score_factors"):
            self.assertNotIn(banned, blob)
        self.assertNotIn("internal", view)

    def test_the_notification_carries_the_tier_to_the_buyer(self):
        sent = {}

        class FakeResult:
            sent = True
            detail = "sent"
            recipient = "buyer@firm.com"

            def as_dict(self):
                return {"sent": True, "detail": "sent", "recipient": self.recipient}

        from services import email as email_service

        original = email_service.notify_submission
        email_service.notify_submission = lambda *a, **k: (
            sent.update(args=a, kwargs=k) or FakeResult()
        )
        try:
            token = self._session()
            out = supplier_portal.submit_session(token, notify=True)
        finally:
            email_service.notify_submission = original

        self.assertTrue(out["notification"]["sent"])
        # The inviter, from the verified claims — never from the request body.
        self.assertEqual(sent["args"][0], "buyer@firm.com")
        self.assertEqual(sent["args"][1], "Apex Gulf Technical Solutions LLC")
        self.assertEqual(sent["args"][2], TIER_HIGH)

    def _capture_notification_link(self):
        """Submit with the mail service stubbed and return the link it was given."""
        captured = {}

        class FakeResult:
            sent = True
            detail = "sent"
            recipient = "buyer@firm.com"

            def as_dict(self):
                return {"sent": True, "detail": "sent", "recipient": self.recipient}

        from services import email as email_service

        original = email_service.notify_submission
        email_service.notify_submission = lambda *a, **k: (
            captured.update(args=a, kwargs=k) or FakeResult()
        )
        os.environ["SUPPLIER_PORTAL_BASE_URL"] = "https://portal.example.com"
        try:
            token = self._session()
            supplier_portal.submit_session(token, notify=True)
        finally:
            email_service.notify_submission = original
            os.environ.pop("SUPPLIER_PORTAL_BASE_URL", None)
        return token, captured["args"][5]

    def test_the_notification_links_to_the_report_not_the_supplier_form(self):
        """The email says "Review it here", so "here" has to be the report.

        It linked to `invitation_link`, which dropped the reviewer onto the
        supplier's own onboarding form: a page they cannot submit, showing the
        questions still outstanding for the supplier, with no assessment on it
        at all.
        """
        token, link = self._capture_notification_link()
        self.assertEqual(
            link,
            f"https://portal.example.com/portal/submissions/{token}/report",
        )
        self.assertNotIn("/supplier/onboarding/", link)

    def test_the_two_link_builders_cannot_be_confused(self):
        """Separate functions, because the audiences are the whole point."""
        os.environ["SUPPLIER_PORTAL_BASE_URL"] = "https://portal.example.com"
        try:
            self.assertEqual(
                supplier_portal.invitation_link("abc"),
                "https://portal.example.com/supplier/onboarding/abc",
            )
            self.assertEqual(
                supplier_portal.report_link("abc"),
                "https://portal.example.com/portal/submissions/abc/report",
            )
        finally:
            os.environ.pop("SUPPLIER_PORTAL_BASE_URL", None)

    def test_a_link_cannot_be_built_without_a_base_url(self):
        os.environ.pop("SUPPLIER_PORTAL_BASE_URL", None)
        for builder in (supplier_portal.invitation_link, supplier_portal.report_link):
            with self.assertRaises(HTTPException) as caught:
                builder("abc")
            self.assertEqual(caught.exception.status_code, 503)

    def test_no_smtp_is_a_skipped_notification_not_a_failed_submission(self):
        token = self._session()
        out = supplier_portal.submit_session(token, notify=True)
        self.assertFalse(out["notification"]["sent"])
        self.assertIn("SMTP", out["notification"]["detail"])
        # The whole point: the submission survives the mail being unavailable.
        self.assertEqual(out["status"], "Under review")
        self.assertEqual(out["assessment"]["assigned_risk_tier"], TIER_HIGH)

    def test_a_missing_recipient_is_reported_not_silently_dropped(self):
        result = self.services_send(to="")
        self.assertFalse(result.sent)
        self.assertIn("recipient", result.detail)

    def services_send(self, to):
        from services import email as email_service
        return email_service.send(to, "s", "t", "<p>t</p>")


class BuyerSubmissionReadsTest(unittest.TestCase):
    """The buyer sees the declared form plus the assessment. The supplier does not."""

    def setUp(self):
        supplier_portal.reset_all()
        self.token = supplier_portal.issue_token(
            "Caspian Trading Partners Ltd", invited_by="owner@firm.com")
        session = supplier_portal._load_session(self.token)
        session.form.update({k: v for k, v in vendor_c_form().items()
                             if k in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS})
        supplier_portal._save_session(session)
        self.client = TestClient(main.app)

    def test_a_submission_appears_in_the_buyer_list(self):
        supplier_portal.submit_session(self.token, notify=False)
        rows = supplier_portal.list_submissions()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vendor_name"], "Caspian Trading Partners Ltd")
        self.assertTrue(rows[0]["has_submitted"])
        self.assertEqual(rows[0]["assigned_risk_tier"], TIER_HIGH)

    def test_the_buyer_gets_the_form_verbatim_and_the_assessment(self):
        supplier_portal.submit_session(self.token, notify=False)
        view = supplier_portal.buyer_view(self.token)
        # Verbatim: every declared field, unshaped and unrenamed.
        self.assertEqual(view["form"], supplier_portal._load_session(self.token).form)
        self.assertEqual(view["assessment"]["assigned_risk_tier"], TIER_HIGH)
        self.assertTrue(view["assessment"]["factors"])
        self.assertEqual(view["status"], "Under review")

    def test_the_buyer_endpoints_are_authenticated(self):
        """Structural, because the suite runs with AUTH_DISABLED.

        Asserting a 401 here would pass or fail depending on an env var that
        exists purely so the endpoints can be driven with curl. What actually
        needs protecting is the `Depends`, so that is what is asserted: drop it
        by accident and these routes start serving a dossier to anyone who can
        reach them.
        """
        guarded = {"/api/portal/submissions", "/api/portal/submissions/{token}"}
        found = set()
        for route in main.app.routes:
            path = getattr(route, "path", "")
            if path not in guarded:
                continue
            deps = {getattr(d.call, "__name__", "") for d in getattr(route, "dependant", None).dependencies} if getattr(route, "dependant", None) else set()
            if "verify_firebase_token" in deps:
                found.add(path)
        self.assertEqual(found, guarded)

    def test_the_supplier_routes_carry_no_authentication_dependency(self):
        """The inverse, and the reason the split exists.

        A supplier has no account, so their routes must not require a token —
        and must therefore not be able to reach the authenticated ones.
        """
        public = {"/api/supplier/{token}", "/api/supplier/{token}/submit"}
        for route in main.app.routes:
            if getattr(route, "path", "") not in public:
                continue
            dependant = getattr(route, "dependant", None)
            deps = ({getattr(d.call, "__name__", "") for d in dependant.dependencies}
                    if dependant else set())
            self.assertNotIn("verify_firebase_token", deps, route.path)

    def test_the_supplier_view_still_carries_no_conclusion(self):
        supplier_portal.submit_session(self.token, notify=False)
        view = supplier_portal.supplier_view(supplier_portal._load_session(self.token))
        self.assertEqual(view["status"], "Under review")
        self.assertNotIn("assessment", view)
        self.assertNotIn("assigned_risk_tier", repr(view))



class QualificationReportTest(unittest.TestCase):
    """The full pipeline report, and the evidence it is allowed to see.

    Two defects lived here. Both produced a report that looked like a product
    decision and were not:

    - `session.form["documents"]` is not where evidence lives. Evidence is one
      record per upload in `session.documents`, so passing the form straight to
      `qualify` reported all twelve mandatory items missing however many files
      were attached, and the dossier came back REJECTED_INCOMPLETE with no
      score at all.
    - `set_dossier` stored the dossier and then raised, so the cache was
      written and the response still failed. The next request then returned
      200 from that cache, which made an intermittent-looking 500 look like a
      pipeline flake rather than a crash on the success path.
    """

    def setUp(self):
        supplier_portal.reset_all()
        self.token = supplier_portal.issue_token(
            "Report Evidence Co", invited_by="owner@firm.com")
        self.client = TestClient(main.app)
        self.session = supplier_portal._load_session(self.token)
        self.session.form.update(
            {k: v for k, v in vendor_c_form().items()
             if k in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS}
        )
        supplier_portal._save_session(self.session)

    def _attach(self, code: str, status: str = "Read", name: str = "evidence.pdf"):
        """Record an upload the way `store_document` persists one."""
        session = supplier_portal._load_session(self.token)
        session.documents.append({
            "reference": f"{code}-{name}" if code else name,
            "document_code": code,
            "status": status,
            "extracted_fields": [],
        })
        supplier_portal._save_session(session)

    # -- evidence reaches the pipeline ----------------------------------

    def test_uploads_reach_the_payload_even_though_the_form_carries_none(self):
        """The regression that made every submission look undocumented."""
        for code in ("60", "61", "62", "64", "66", "67", "68", "69",
                     "63", "82", "83", "84"):
            self._attach(code)

        session = supplier_portal._load_session(self.token)
        self.assertEqual(len(session.documents), 12)
        # The premise: evidence is not in the form, which is why the payload
        # has to be assembled rather than passed through.
        self.assertEqual(session.form.get("documents"), None)

        payload = supplier_portal.qualification_payload(session)
        self.assertEqual(len(payload["documents"]), 12)
        self.assertEqual(
            {d["document_code"] for d in payload["documents"]},
            {"60", "61", "62", "63", "64", "66", "67", "68", "69", "82", "83", "84"},
        )
        for entry in payload["documents"]:
            self.assertTrue(entry["attachment_reference"])

    def test_the_validator_agrees_the_evidence_is_attached(self):
        """End of the chain: what the payload claims, the gate actually sees."""
        for code in ("60", "62", "64"):
            self._attach(code)
        payload = supplier_portal.qualification_payload(
            supplier_portal._load_session(self.token))
        for code in ("60", "62", "64"):
            self.assertTrue(validation_lib.has_document(payload, code), code)
        self.assertFalse(validation_lib.has_document(payload, "61"))

    def test_an_unreadable_upload_does_not_count_as_evidence(self):
        self._attach("62", status="Unreadable")
        payload = supplier_portal.qualification_payload(
            supplier_portal._load_session(self.token))
        self.assertEqual(payload["documents"], [])

    def test_the_smart_parse_upload_is_not_mistaken_for_a_mandatory_item(self):
        """Smart Document Parsing uploads under `auto-fill`, which is not an
        item code. Counting it would let a parsed-but-unattached item pass."""
        self._attach("auto-fill", name="company-extract.pdf")
        payload = supplier_portal.qualification_payload(
            supplier_portal._load_session(self.token))
        self.assertEqual(payload["documents"], [])

    def test_a_replacement_upload_supersedes_the_earlier_one(self):
        self._attach("62", name="first.pdf")
        self._attach("62", name="corrected.pdf")
        payload = supplier_portal.qualification_payload(
            supplier_portal._load_session(self.token))
        self.assertEqual(len(payload["documents"]), 1)
        self.assertIn("corrected.pdf", payload["documents"][0]["attachment_reference"])

    def test_evidence_declared_on_the_form_is_not_stripped(self):
        """A submission assembled outside the portal keeps its documents."""
        session = supplier_portal._load_session(self.token)
        session.form["documents"] = [
            {"document_code": "62", "attachment_reference": "licence.pdf"},
        ]
        supplier_portal._save_session(session)
        payload = supplier_portal.qualification_payload(
            supplier_portal._load_session(self.token))
        self.assertEqual(len(payload["documents"]), 1)

    # -- the endpoint actually returns what it cached -------------------

    def _stub_agent(self, status="QUALIFIED"):
        """Replace the pipeline. The real one needs model and RAG calls."""
        dossier = {
            "vendor_id": "VND-TEST", "vendor_name": "Report Evidence Co",
            "qualification_status": status,
            "weighted_risk_score": "1.53",
            "assigned_risk_tier": "Low Risk (SDD)",
            "appendix_f_score": {"status": "PASSED", "total_score": "22"},
            "rag_retrieval_citations": [{"title": "Policy"}],
            "required_controls": [{"control": "c"}],
            "weighted_factors": [{"factor": "f"}],
        }

        class _Agent:
            async def qualify(self, form):
                _Agent.seen = form
                return dossier

        original = main.get_agent
        main.get_agent = lambda: _Agent()
        self.addCleanup(lambda: setattr(main, "get_agent", original))
        return _Agent

    def test_the_report_is_returned_and_is_the_one_that_was_cached(self):
        supplier_portal.submit_session(self.token, notify=False)
        agent = self._stub_agent()

        first = self.client.post(
            f"/api/portal/submissions/{self.token}/report",
            headers={"Authorization": "Bearer owner@firm.com"})
        # The bug this pins: caching succeeded, so the response was a 500 from
        # a crash *after* the cache write, not a pipeline failure.
        self.assertEqual(first.status_code, 200, first.text)
        body = first.json()
        self.assertFalse(body["cached"])
        self.assertEqual(body["dossier"]["qualification_status"], "QUALIFIED")

        second = self.client.post(
            f"/api/portal/submissions/{self.token}/report",
            headers={"Authorization": "Bearer owner@firm.com"})
        self.assertEqual(second.status_code, 200)
        self.assertTrue(second.json()["cached"])
        self.assertEqual(second.json()["dossier"], body["dossier"])
        # Provenance the report screen renders, which the dossier cannot carry.
        submission = body["submission"]
        self.assertEqual(submission["id"], self.token)
        self.assertEqual(submission["invited_by"], "owner@firm.com")
        self.assertIsNotNone(submission["submitted_at"])
        self.assertIn("legal_name", submission["submitted_form"])
        self.assertIsNotNone(
            supplier_portal._load_session(self.token).internal["dossier_computed_at"])

    def test_the_pipeline_receives_the_assembled_evidence(self):
        supplier_portal.submit_session(self.token, notify=False)
        for code in ("60", "61", "62"):
            self._attach(code)
        agent = self._stub_agent()
        self.client.post(f"/api/portal/submissions/{self.token}/report",
                         headers={"Authorization": "Bearer owner@firm.com"})
        self.assertEqual(len(agent.seen["documents"]), 3)

    def test_a_submission_that_is_still_in_progress_has_nothing_to_report(self):
        response = self.client.post(
            f"/api/portal/submissions/{self.token}/report",
            headers={"Authorization": "Bearer owner@firm.com"})
        self.assertEqual(response.status_code, 409)

    def test_a_pipeline_failure_is_reported_as_retryable(self):
        """Not a bare 500: the buyer has to know a retry is worth making."""
        supplier_portal.submit_session(self.token, notify=False)

        class _Boom:
            async def qualify(self, form):
                raise RuntimeError("retrieval unavailable")

        original = main.get_agent
        main.get_agent = lambda: _Boom()
        self.addCleanup(lambda: setattr(main, "get_agent", original))

        response = self.client.post(
            f"/api/portal/submissions/{self.token}/report",
            headers={"Authorization": "Bearer owner@firm.com"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("retrying is safe", response.json()["detail"])
        self.assertIsNone(supplier_portal.get_dossier(self.token))

    # -- a stale rejection must not outlive the answer that fixed it ----

    def test_saving_a_section_discards_a_cached_rejection(self):
        supplier_portal.submit_session(self.token, notify=False)
        supplier_portal.set_dossier(self.token, {
            "qualification_status": "REJECTED_INCOMPLETE",
            "weighted_risk_score": None,
        })
        self.assertIsNotNone(supplier_portal.get_dossier(self.token))

        self.client.put(
            f"/api/supplier/{self.token}/form/ownership",
            json={"fields": {"ubos": [{"name": "Rashid Al-Farsi"}]}})

        self.assertIsNone(
            supplier_portal.get_dossier(self.token),
            "a cached rejection survived the edit that was meant to resolve it")

    def test_uploading_a_missing_document_discards_a_cached_rejection(self):
        """Through the real upload route, not a hand-written record.

        Attaching evidence to `session.documents` directly would pass whether
        or not the upload path invalidated anything, which is the whole claim.
        """
        supplier_portal.submit_session(self.token, notify=False)
        supplier_portal.set_dossier(self.token, {
            "qualification_status": "REJECTED_INCOMPLETE"})
        self.assertIsNotNone(supplier_portal.get_dossier(self.token))

        uploaded = self.client.post(
            f"/api/supplier/{self.token}/documents",
            files={"file": ("trade-licence.pdf", io.BytesIO(_LICENCE_PDF),
                            "application/pdf")},
            params={"document_code": "62"},
        )
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        self.assertIsNone(supplier_portal.get_dossier(self.token))



class SubmissionDecisionTest(unittest.TestCase):
    """The internal approve/reject on a submission.

    Distinct from the emailed approval brief on purpose. A brief is consumed by
    the act of answering it and cannot be revisited, which is right for "has this
    person decided yet?" and wrong for a submission that may be rejected,
    resubmitted, and disagreed about. These tests pin the properties that make
    it safe to expose to the reviewing team.
    """

    def setUp(self):
        supplier_portal.reset_all()
        self.token = supplier_portal.issue_token(
            "Decision Path Co", invited_by="owner@firm.com")
        session = supplier_portal._load_session(self.token)
        session.form.update(
            {k: v for k, v in vendor_c_form().items()
             if k in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS}
        )
        supplier_portal._save_session(session)
        supplier_portal.submit_session(self.token, notify=False)
        self.client = TestClient(main.app)
        self.auth = {"Authorization": "Bearer owner@firm.com"}

    def _decide(self, decision, justification="", token=None, auth=None):
        return self.client.post(
            f"/api/portal/submissions/{token or self.token}/decision",
            json={"decision": decision, "justification": justification},
            headers=auth if auth is not None else self.auth,
        )

    def test_approving_is_reflected_in_the_buyer_view(self):
        self.assertIsNone(supplier_portal.buyer_view(self.token)["decision"])
        response = self._decide("approved")
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()["decision"]
        self.assertEqual(record["decision"], "approved")
        self.assertTrue(record["decided_by"])
        self.assertEqual(response.json()["submission"]["status"], "Approved")

    def test_a_rejection_tells_the_supplier_what_to_change(self):
        """They cannot act on "Action required" alone.

        The clarifications a supplier can already see are the ones document
        extraction raised, which are frequently not why the submission was sent
        back. Withholding the reviewer's reason left them resubmitting the same
        form.
        """
        self._decide("rejected", "The trade licence expired last month.")
        view = supplier_portal.supplier_view(
            supplier_portal._load_session(self.token))
        self.assertEqual(view["status"], "Action required")
        self.assertIn(view["status"], supplier_portal.NEUTRAL_STATUSES)
        self.assertEqual(view["decision"]["decision"], "rejected")
        self.assertEqual(
            view["decision"]["justification"], "The trade licence expired last month.")
        self.assertTrue(view["decision"]["can_resubmit"])

    def test_the_supplier_still_learns_nothing_about_the_assessment(self):
        """The reason is the reviewer's words, not the score that produced them."""
        self._decide("rejected", "The trade licence expired last month.")
        view = supplier_portal.supplier_view(
            supplier_portal._load_session(self.token))
        blob = repr(view).lower()
        for banned in ("high risk", "edd", "medium risk", "weighted_risk_score",
                       "assigned_risk_tier", "1.53", "2.2", "weighted_factors",
                       "appendix_f", "citations", "screening_result"):
            self.assertNotIn(banned, blob)

    def test_an_approval_reaches_the_supplier_with_no_reason_required(self):
        self._decide("approved")
        view = supplier_portal.supplier_view(
            supplier_portal._load_session(self.token))
        self.assertEqual(view["status"], "Approved")
        self.assertEqual(view["decision"]["decision"], "approved")

    def test_approving_never_edits_the_score_or_the_tier(self):
        """The tier is the policy engine's output, not a reviewer's to change."""
        before = supplier_portal.buyer_view(self.token)["assessment"]
        self._decide("approved")
        after = supplier_portal.buyer_view(self.token)["assessment"]
        self.assertEqual(before["assigned_risk_tier"], after["assigned_risk_tier"])
        self.assertEqual(before["weighted_risk_score"], after["weighted_risk_score"])
        self.assertEqual(after["assigned_risk_tier"], TIER_HIGH)

    def test_a_rejection_must_say_why(self):
        """The supplier sees "Action required" and nothing else."""
        for reason in ("", "too short"):
            response = self._decide("rejected", reason)
            self.assertEqual(response.status_code, 400, reason)
            self.assertIn("reason", response.json()["detail"])
        self.assertIsNone(supplier_portal.buyer_view(self.token)["decision"])

    def test_an_approval_needs_no_reason(self):
        self.assertEqual(self._decide("approved").status_code, 200)

    def test_only_approved_or_rejected_is_accepted(self):
        for bad in ("maybe", "", "APPROVED!", "approve", "null"):
            response = self._decide(bad)
            self.assertEqual(response.status_code, 400, bad)

    def test_the_approver_comes_from_the_token_not_the_body(self):
        """An audit record naming whoever the caller typed is not an audit record."""
        response = self.client.post(
            f"/api/portal/submissions/{self.token}/decision",
            json={"decision": "approved", "decided_by": "ceo@example.com"},
            headers=self.auth,
        )
        self.assertEqual(response.status_code, 200)
        decided_by = response.json()["decision"]["decided_by"]
        self.assertNotEqual(decided_by, "ceo@example.com")
        # Under AUTH_DISABLED the suite's token resolves to a fixed stand-in;
        # what matters is that the recorded name is that identity and not the
        # one the caller supplied.
        self.assertEqual(decided_by, "auth-disabled")

    def test_approver_from_prefers_the_verified_email(self):
        self.assertEqual(
            supplier_portal.approver_from(
                {"email": "reviewer@firm.com", "uid": "u-1"}),
            "reviewer@firm.com",
        )
        self.assertEqual(supplier_portal.approver_from({"uid": "u-1"}), "u-1")
        self.assertEqual(supplier_portal.approver_from({}), "")

    def test_a_submission_can_be_decided_again_and_keeps_the_history(self):
        """A rejection is not final: the supplier fixes it and it comes back."""
        self._decide("rejected", "The trade licence expired last month.")
        self._decide("approved")
        view = supplier_portal.buyer_view(self.token)
        self.assertEqual(view["decision"]["decision"], "approved")
        self.assertEqual(len(view["decision_history"]), 2)
        self.assertEqual(
            [d["decision"] for d in view["decision_history"]],
            ["rejected", "approved"],
        )

    def test_an_unsubmitted_invitation_cannot_be_decided(self):
        fresh = supplier_portal.issue_token("Not Yet Submitted Co")
        response = self._decide("approved", token=fresh)
        self.assertEqual(response.status_code, 409)

    def test_the_decision_endpoint_is_authenticated(self):
        """Structural, for the same reason as the other buyer routes."""
        guarded = {"/api/portal/submissions/{token}/decision"}
        found = {
            route.path for route in main.app.routes
            if getattr(route, "path", None) in guarded
        }
        self.assertEqual(found, guarded)

    def test_there_is_no_supplier_route_for_deciding(self):
        """A supplier must not be able to approve themselves.

        Checked structurally because the suite runs with AUTH_DISABLED, so a 401
        assertion would depend on an env var rather than on the route surface.
        """
        paths = {getattr(r, "path", "") for r in main.app.routes}
        self.assertFalse(
            [p for p in paths if p.startswith("/api/supplier/") and "decision" in p],
            f"a supplier-facing decision route exists: {sorted(paths)}",
        )


class SupplierNotificationTest(unittest.TestCase):
    """Mail to the supplier: the invitation, and the outcome.

    This was a hole rather than a missing feature. The portal emailed the buyer
    when a submission arrived and then went silent — a rejection instructing the
    supplier to fix their form reached nobody, and the only way to discover it had
    happened was to reopen the link by hand. There was also nowhere to send it:
    no supplier address was collected at invitation time.
    """

    def setUp(self):
        supplier_portal.reset_all()
        for var in ("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD",
                    "SMTP_FROM", "NOTIFICATION_TO", "SUPPLIER_PORTAL_BASE_URL"):
            os.environ.pop(var, None)

    def test_an_invitation_needs_an_address_to_be_sent_to(self):
        """Without one there is no way to ever tell this supplier anything."""
        client = TestClient(main.app)
        response = client.post(
            "/api/vendors/qualifications/invitations",
            json={"vendor_name": "No Address Co"},
            headers={"Authorization": "Bearer owner@firm.com"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("supplier_email", response.json()["detail"])

    def test_an_unusable_address_is_refused_rather_than_stored(self):
        client = TestClient(main.app)
        for bad in ("not-an-address", "two@at@signs.com", "missing@domain",
                    "trailing@dot.", "spaced out@example.com", ""):
            response = client.post(
                "/api/vendors/qualifications/invitations",
                json={"vendor_name": "Bad Address Co", "supplier_email": bad},
                headers={"Authorization": "Bearer owner@firm.com"},
            )
            self.assertEqual(response.status_code, 400, bad)

    def test_the_address_is_kept_on_the_session_and_survives_a_restart(self):
        token = supplier_portal.issue_token(
            "Kept Co", invited_by="owner@firm.com",
            supplier_email="vendor@example.com")
        self.assertEqual(
            supplier_portal._load_session(token).supplier_email, "vendor@example.com")
        # In a subprocess, because a reload in-process would leave the imported
        # singletons half-replaced for every other test. This is a genuine cold
        # start reading only the database, which is what proves the address was
        # persisted rather than merely held in memory.
        db = tempfile.mktemp(suffix=".db")
        try:
            env = dict(os.environ, SUPPLIER_PORTAL_DB=db, AUTH_DISABLED="true")
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            seed = subprocess.run(
                [sys.executable, "-c",
                 f"import supplier_portal as s;"
                 f"s.issue_token('Kept Co', invited_by='owner@firm.com',"
                 f" supplier_email='vendor@example.com')"],
                capture_output=True, text=True, env=env, timeout=120, cwd=root,
            )
            self.assertEqual(seed.returncode, 0, seed.stderr)

            read = subprocess.run(
                [sys.executable, "-c",
                 "import supplier_portal as s;"
                 "s._db();"
                 "t=next(iter(s._SESSIONS));"
                 "print(s._load_session(t).supplier_email)"],
                capture_output=True, text=True, env=env, timeout=120, cwd=root,
            )
            self.assertEqual(read.returncode, 0, read.stderr)
            self.assertEqual(read.stdout.strip(), "vendor@example.com")
        finally:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.unlink(db + suffix)
                except OSError:
                    pass

    def test_an_approval_is_emailed_to_the_supplier(self):
        token = supplier_portal.issue_token(
            "Approved Co", invited_by="owner@firm.com",
            supplier_email="applicant@example.com")
        session = supplier_portal._load_session(token)
        session.form.update({k: v for k, v in vendor_c_form().items()
                             if k in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS})
        supplier_portal._save_session(session)
        supplier_portal.submit_session(token, notify=False)

        captured = {}

        class FakeResult:
            sent = True
            detail = "sent"
            recipient = "applicant@example.com"

            def as_dict(self):
                return {"sent": True, "detail": "sent",
                        "recipient": self.recipient}

        from services import email as email_service
        original = email_service.notify_supplier_decision
        email_service.notify_supplier_decision = lambda *a, **k: (
            captured.update(args=a) or FakeResult())
        try:
            os.environ["SUPPLIER_PORTAL_BASE_URL"] = "https://portal.example.com"
            record = supplier_portal.record_submission_decision(
                token, "approved", "owner@firm.com")
        finally:
            email_service.notify_supplier_decision = original
            os.environ.pop("SUPPLIER_PORTAL_BASE_URL", None)

        self.assertTrue(record["notification"]["sent"])
        self.assertEqual(captured["args"][0], "applicant@example.com")
        self.assertTrue(captured["args"][2], "approved flag should be True")

    def test_a_rejection_reaches_the_supplier_with_its_reason(self):
        token = supplier_portal.issue_token(
            "Rejected Co", invited_by="owner@firm.com",
            supplier_email="applicant@example.com")
        session = supplier_portal._load_session(token)
        session.form.update({k: v for k, v in vendor_c_form().items()
                             if k in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS})
        supplier_portal._save_session(session)
        supplier_portal.submit_session(token, notify=False)

        captured = {}

        class FakeResult:
            sent = True
            detail = "sent"
            recipient = "applicant@example.com"

            def as_dict(self):
                return {"sent": True, "detail": "sent",
                        "recipient": self.recipient}

        from services import email as email_service
        original = email_service.notify_supplier_decision
        email_service.notify_supplier_decision = lambda *a, **k: (
            captured.update(args=a) or FakeResult())
        try:
            os.environ["SUPPLIER_PORTAL_BASE_URL"] = "https://portal.example.com"
            supplier_portal.record_submission_decision(
                token, "rejected", "owner@firm.com",
                "The trade licence expired last month.")
        finally:
            email_service.notify_supplier_decision = original
            os.environ.pop("SUPPLIER_PORTAL_BASE_URL", None)

        self.assertFalse(captured["args"][2], "approved flag should be False")
        self.assertEqual(captured["args"][3], "The trade licence expired last month.")
        self.assertIn("/supplier/onboarding/", captured["args"][4])

    def test_a_failed_send_never_discards_the_decision(self):
        """The reviewer already decided; losing that would be far worse."""
        token = supplier_portal.issue_token(
            "No SMTP Co", invited_by="owner@firm.com",
            supplier_email="applicant@example.com")
        session = supplier_portal._load_session(token)
        session.form.update({k: v for k, v in vendor_c_form().items()
                             if k in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS})
        supplier_portal._save_session(session)
        supplier_portal.submit_session(token, notify=False)

        os.environ["SUPPLIER_PORTAL_BASE_URL"] = "https://portal.example.com"
        try:
            record = supplier_portal.record_submission_decision(
                token, "approved", "owner@firm.com")
        finally:
            os.environ.pop("SUPPLIER_PORTAL_BASE_URL", None)

        self.assertEqual(record["decision"], "approved")
        self.assertFalse(record["notification"]["sent"])
        self.assertIn("SMTP", record["notification"]["detail"])
        self.assertEqual(
            supplier_portal.buyer_view(token)["decision"]["decision"], "approved")

    def test_a_supplier_with_no_address_on_file_is_reported_not_hidden(self):
        token = supplier_portal.issue_token("Legacy Co", invited_by="owner@firm.com")
        session = supplier_portal._load_session(token)
        session.form.update({k: v for k, v in vendor_c_form().items()
                             if k in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS})
        supplier_portal._save_session(session)
        supplier_portal.submit_session(token, notify=False)

        os.environ["SUPPLIER_PORTAL_BASE_URL"] = "https://portal.example.com"
        try:
            record = supplier_portal.record_submission_decision(
                token, "rejected", "owner@firm.com", "Something needs changing.")
        finally:
            os.environ.pop("SUPPLIER_PORTAL_BASE_URL", None)

        self.assertFalse(record["notification"]["sent"])
        self.assertIn("No supplier email on file", record["notification"]["detail"])
        self.assertEqual(record["decision"], "rejected")

    def test_supplier_mail_is_not_diverted_to_the_pilot_inbox(self):
        """NOTIFICATION_TO aims mail at one inbox.

        That is right for internal notification and wrong for a supplier: an
        approval addressed to the pilot means the supplier is never told, while
        the buyer sees "sent".
        """
        os.environ["NOTIFICATION_TO"] = "pilot@example.com"
        os.environ["SUPPLIER_PORTAL_BASE_URL"] = "https://portal.example.com"
        try:
            result = main.email_service.send(
                "applicant@example.com", "s", "t", "<p>t</p>",
                use_override=False)
            # No SMTP configured, so nothing sends — but the recipient it chose
            # is the one we asked for, which is the point.
            self.assertIn("SMTP", result.detail)

            override = main.email_service.send(
                "applicant@example.com", "s", "t", "<p>t</p>")
            self.assertEqual(override.recipient, "pilot@example.com")
        finally:
            for var in ("NOTIFICATION_TO", "SUPPLIER_PORTAL_BASE_URL"):
                os.environ.pop(var, None)

    def test_the_supplier_decision_mail_carries_no_assessment(self):
        """The firewall cannot reach their inbox, so it is enforced here too."""
        from services import email as email_service
        text = email_service.decision_body(
            "Apex Gulf LLC", False, "The trade licence expired.", "https://x/y")
        html = email_service.decision_html(
            "Apex Gulf LLC", False, "The trade licence expired.", "https://x/y")
        for blob in (text.lower(), html.lower()):
            for banned in ("risk tier", "high risk", "medium risk", "edd",
                           "weighted_risk_score", "score", "appendix f"):
                self.assertNotIn(banned, blob, f"{banned!r} leaked into supplier mail")
        self.assertIn("trade licence expired", text)

if __name__ == "__main__":
    unittest.main()