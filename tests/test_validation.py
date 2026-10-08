"""The Zero-Hallucination gate: what blocks, what does not, and why."""

import unittest

import risk_engine
import signals
import validation
from agents.vendor_qualification.agent import onboarding_flags

from tests.fixtures import (
    ALL_SUPPLIER_DOC_CODES,
    complete_form,
    with_documents,
    without,
    without_extra_evidence,
)


def _missing_fields(form) -> set:
    return {m.field for m in validation.validate(form).missing}


def _document_gaps(form) -> list:
    """The human-readable attachment gaps the validator reports."""
    for gap in validation.validate(form).missing:
        if gap.details:
            return gap.details
    return []


def _score_of(form) -> tuple:
    """The deterministic score and tier, for asserting nothing moved them."""
    assessment = risk_engine.assess(signals.build_signals(form))
    return assessment.weighted_risk_score, assessment.assigned_risk_tier


def _missing_reason(form, field: str) -> str:
    """The human-readable reason the gate gives for one field."""
    for gap in validation.validate(form).missing:
        if gap.field == field:
            return gap.reason or ""
    return ""


class CompleteSubmissionTest(unittest.TestCase):
    def test_the_reference_submission_passes(self):
        report = validation.validate(complete_form())
        self.assertTrue(
            report.is_complete,
            msg=f"unexpected gaps: {[m.to_dict() for m in report.missing]}",
        )
        self.assertEqual(report.missing, [])

    def test_an_empty_submission_reports_every_blocking_gap(self):
        report = validation.validate({})
        self.assertFalse(report.is_complete)
        self.assertGreater(len(report.missing), 20)
        # Trade licence, turnover and UBO details are all reported by name.
        for field in ("legal_name", "trade_license_no", "turnover_year_1", "ubos"):
            self.assertIn(field, {m.field for m in report.missing})

    def test_a_non_dict_body_is_treated_as_empty(self):
        self.assertFalse(validation.validate(None).is_complete)
        self.assertFalse(validation.validate("nope").is_complete)


class MissingValueSemanticsTest(unittest.TestCase):
    def test_na_is_accepted_only_where_the_form_allows_it(self):
        # Item 35 instructs "State N/A if none", so a literal N/A satisfies it.
        form = complete_form()
        form["ultimate_parent_company"] = "N/A"
        self.assertNotIn("ultimate_parent_company", _missing_fields(form))
        # The trade licence has no N/A option, so the same answer does nothing.
        licence = complete_form()
        licence["trade_license_no"] = "N/A"
        self.assertIn("trade_license_no", _missing_fields(licence))

    def test_an_na_trade_licence_is_treated_as_not_supplied(self):
        form = complete_form()
        form["trade_license_no"] = "N/A"
        self.assertIn("trade_license_no", _missing_fields(form))

    def test_zero_and_blank_are_missing(self):
        form = complete_form()
        form["turnover_year_1"] = 0
        self.assertIn("turnover_year_1", _missing_fields(form))
        form = complete_form()
        form["legal_name"] = "   "
        self.assertIn("legal_name", _missing_fields(form))

    def test_a_declared_no_is_a_supplied_answer(self):
        form = complete_form()
        form["regulatory_history_declaration"] = "no"
        self.assertNotIn("regulatory_history_declaration", _missing_fields(form))

    def test_an_unanswered_declaration_blocks(self):
        for blank in ("", "N/A", "-", None):
            with self.subTest(blank=blank):
                form = complete_form()
                form["intermediary_declaration"] = blank
                self.assertIn("intermediary_declaration", _missing_fields(form))

    def test_a_malformed_year_blocks(self):
        form = complete_form()
        form["year_of_commencement"] = "since 2015"
        self.assertIn("year_of_commencement", _missing_fields(form))

    def test_a_short_value_fails_its_minimum_length(self):
        form = complete_form()
        form["goods_services_proposed"] = "repairs"
        self.assertIn("goods_services_proposed", _missing_fields(form))


class TableTest(unittest.TestCase):
    def test_three_client_references_are_required(self):
        form = complete_form()
        form["client_references"] = form["client_references"][:2]
        self.assertIn("client_references", _missing_fields(form))

    def test_an_incomplete_reference_row_blocks(self):
        form = complete_form()
        form["client_references"][0].pop("contact_details")
        self.assertIn("client_references", _missing_fields(form))

    def test_empty_table_reports_zero_found(self):
        # The min-count message used to quote the requirement as the count, so
        # an empty table read "at least 3 entries are required (3 found)".
        form = complete_form()
        form["client_references"] = []
        problem = _missing_reason(form, "client_references")
        self.assertIn("0 found", problem)
        self.assertNotIn("3 found", problem)

    def test_short_table_reports_the_actual_count(self):
        form = complete_form()
        form["client_references"] = form["client_references"][:2]
        self.assertIn("2 found", _missing_reason(form, "client_references"))

    def test_a_ubo_row_without_a_percentage_blocks(self):
        form = complete_form()
        form["ubos"][0].pop("ownership_percentage")
        self.assertIn("ubos", _missing_fields(form))


class DocumentTest(unittest.TestCase):
    def test_all_every_supplier_document_is_required(self):
        for code in ALL_SUPPLIER_DOC_CODES:
            with self.subTest(document=code):
                form = with_documents(complete_form(), [c for c in ALL_SUPPLIER_DOC_CODES if c != code])
                self.assertIn("documents", _missing_fields(form))

    def test_the_vat_certificate_blocks_only_for_a_vat_registered_supplier(self):
        codes = [c for c in ("60", "61", "62", "64", "66", "67", "68", "69", "73")]
        registered = with_documents(without_extra_evidence(complete_form()), codes)
        self.assertIn("documents", _missing_fields(registered))

        not_registered = without_extra_evidence(complete_form())
        not_registered["vat_registration_status"] = "not applicable"
        not_registered.pop("vat_registration_no")
        self.assertNotIn("documents", _missing_fields(with_documents(not_registered, codes)))

    def test_the_bank_confirmation_blocks_only_over_375k(self):
        codes = [c for c in ("60", "61", "62", "64", "66", "67", "68", "69", "63")]

        under = without_extra_evidence(complete_form())
        under["estimated_spend_aed"] = 300_000
        self.assertNotIn("documents", _missing_fields(with_documents(under, codes)))

        over = without_extra_evidence(complete_form())
        over["estimated_spend_aed"] = 375_000
        self.assertIn("documents", _missing_fields(with_documents(over, codes)))

    def test_the_intermediary_agreement_blocks_only_for_a_declared_intermediary(self):
        codes = [c for c in ("60", "61", "62", "64", "66", "67", "68", "69", "63", "73")]

        declared = without_extra_evidence(complete_form())
        declared["is_distributor"] = True
        self.assertIn("documents", _missing_fields(with_documents(declared, codes)))

        self.assertNotIn(
            "documents",
            _missing_fields(with_documents(without_extra_evidence(complete_form()), codes)),
        )


class AdvisoryTest(unittest.TestCase):
    def test_advisory_gaps_are_reported_but_never_block(self):
        form = without(complete_form(), "number_of_employees", "geographic_coverage",
                       "public_review_links", "ownership_structure_chart")
        report = validation.validate(form)
        self.assertTrue(report.is_complete)
        advisory = {m.field for m in report.advisory}
        self.assertEqual(advisory, {
            "number_of_employees", "geographic_coverage",
            "public_review_links", "ownership_structure_chart",
        })


class CertificationTest(unittest.TestCase):
    """Item 20: a structured multi-select, not the free-text box it replaced."""

    def test_an_unanswered_item_20_blocks(self):
        form = without(complete_form(), "certifications")
        self.assertIn("certifications", _missing_fields(form))

    def test_every_option_is_a_valid_answer(self):
        for option in validation.CERTIFICATION_VALUES:
            with self.subTest(option=option):
                form = without_extra_evidence(complete_form())
                form["certifications"] = [option]
                self.assertNotIn("certifications", _missing_fields(form))

    def test_none_alone_asks_for_no_certificate(self):
        form = without(complete_form(), "certifications")
        form["certifications"] = ["none"]
        report = validation.validate(form)
        self.assertTrue(report.is_complete, msg=[m.to_dict() for m in report.missing])
        self.assertNotIn("quality_or_infosec_certified", report.active_conditions)
        self.assertNotIn("esg_certified", report.active_conditions)

    def test_an_unrecognised_option_blocks_rather_than_being_ignored(self):
        # A typo must not read as "no certifications claimed" and quietly
        # release the supplier from the certificate uploads.
        form = complete_form()
        form["certifications"] = ["iso_9001 typo"]
        self.assertIn("certifications", _missing_fields(form))

    def test_iso_9001_or_iso_27001_pulls_in_item_82(self):
        for option in ("iso_9001", "iso_27001_soc2"):
            with self.subTest(option=option):
                form = without(complete_form(), "certifications", "documents")
                form["certifications"] = [option]
                gaps = _document_gaps(form)
                self.assertIn("Item 82 Quality & Information Security Certificates (ISO / SOC 2)",
                              gaps)
                # InfoSec and quality share one slot, so neither adds a second.
                self.assertNotIn("Item 83 ESG / Code of Conduct Compliance Document", gaps)

    def test_esg_pulls_in_item_83_only(self):
        form = without(complete_form(), "certifications", "documents")
        form["certifications"] = ["hse_esg"]
        gaps = _document_gaps(form)
        self.assertIn("Item 83 ESG / Code of Conduct Compliance Document", gaps)
        self.assertNotIn("Item 82 Quality & Information Security Certificates (ISO / SOC 2)",
                         gaps)

    def test_both_certificate_slots_are_demanded_together(self):
        form = without(complete_form(), "certifications", "documents")
        form["certifications"] = ["iso_9001", "iso_27001_soc2", "hse_esg"]
        gaps = _document_gaps(form)
        self.assertIn("Item 82 Quality & Information Security Certificates (ISO / SOC 2)",
                      gaps)
        self.assertIn("Item 83 ESG / Code of Conduct Compliance Document", gaps)

    def test_a_comma_separated_string_is_read_for_legacy_payloads(self):
        form = without(complete_form(), "certifications", "documents")
        form["certifications"] = "iso_9001, hse_esg"
        gaps = _document_gaps(form)
        self.assertIn("Item 82 Quality & Information Security Certificates (ISO / SOC 2)",
                      gaps)
        self.assertIn("Item 83 ESG / Code of Conduct Compliance Document", gaps)


class AuditedFinancialsTest(unittest.TestCase):
    """Item 40a: the declaration, and the Item 84 slot behind a Yes."""

    def test_an_unanswered_declaration_blocks(self):
        form = without(complete_form(), "financial_statements_audited")
        self.assertIn("financial_statements_audited", _missing_fields(form))

    def test_no_is_a_complete_answer(self):
        # Dropping only 82/83/84 leaves the eight base documents in place, so a
        # block here could only be the audit declaration or its certificate slot.
        form = without_extra_evidence(
            without(complete_form(), "financial_statements_audited"))
        form["financial_statements_audited"] = "No"
        self.assertTrue(
            validation.validate(form).is_complete,
            msg=[m.to_dict() for m in validation.validate(form).missing],
        )

    def test_yes_without_the_report_still_blocks(self):
        form = without(complete_form(), "financial_statements_audited", "documents")
        form["financial_statements_audited"] = "Yes"
        self.assertIn("Item 84 Audited Balance Sheet / Financial Report (Last 2 Years)",
                      _document_gaps(form))

    def test_only_a_yes_activates_the_condition(self):
        for answer, active in (("Yes", True), ("No", False), ("no", False)):
            with self.subTest(answer=answer):
                form = without(complete_form(), "financial_statements_audited")
                form["financial_statements_audited"] = answer
                report = validation.validate(form)
                self.assertEqual(
                    "financial_statements_audited" in report.active_conditions, active)


class OnboardingFlagTest(unittest.TestCase):
    """The Tool 2 hand-off. Not a risk input — nothing here moves a score."""

    def test_the_bank_callback_is_always_required(self):
        for form in ({}, complete_form(), {"financial_statements_audited": "No"}):
            with self.subTest(form=sorted(form)[:2]):
                self.assertTrue(
                    onboarding_flags(form)["bank_callback_verification_required"])

    def test_audited_financials_are_verified_only_with_the_declaration_and_the_document(self):
        declared = complete_form()
        self.assertTrue(onboarding_flags(declared)["audited_financials_verified"])

        no_document = without(declared, "documents")
        self.assertFalse(onboarding_flags(no_document)["audited_financials_verified"])

        not_audited = dict(declared, financial_statements_audited="No")
        self.assertFalse(onboarding_flags(not_audited)["audited_financials_verified"])

        unanswered = without(declared, "financial_statements_audited")
        self.assertFalse(onboarding_flags(unanswered)["audited_financials_verified"])

    def test_a_stale_attachment_does_not_verify_unaudited_statements(self):
        # The document is present but the supplier said No. Believing the file
        # over the declaration would assert a verification that never happened.
        form = dict(complete_form(), financial_statements_audited="No")
        self.assertFalse(onboarding_flags(form)["audited_financials_verified"])

    def test_an_empty_submission_never_claims_verified_financials(self):
        self.assertFalse(onboarding_flags({})["audited_financials_verified"])

    def test_the_flags_do_not_move_the_score(self):
        before = _score_of(complete_form())
        audited = complete_form()
        audited["financial_statements_audited"] = "No"
        self.assertEqual(before, _score_of(audited))


class MissingDataRequestTest(unittest.TestCase):
    def test_the_request_is_actionable_and_cites_form_items(self):
        request = validation.missing_data_request(validation.validate({}))
        self.assertEqual(request["count"], len(request["fields"]))
        self.assertIn("NH-PQF-001", request["form"])
        for field in request["fields"]:
            self.assertTrue(field["label"])
            self.assertTrue(field["form_item"].startswith("Item"))
            self.assertTrue(field["reason"])

    def test_no_active_conditions_reported_for_a_clean_form(self):
        report = validation.validate(complete_form())
        # The reference supplier answers "no" to every conditional declaration.
        self.assertNotIn("intermediary_or_reseller", report.active_conditions)
        self.assertNotIn("adverse_matter", report.active_conditions)


if __name__ == "__main__":
    unittest.main()
