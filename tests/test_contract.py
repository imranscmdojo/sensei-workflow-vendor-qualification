"""
Output-contract enforcement: a dossier that lies about its own arithmetic must
be caught.

These tests build real dossiers from the deterministic engine (no model), then
tamper with them one field at a time to prove `contract.audit()` notices.
"""

import unittest
from decimal import Decimal

import contract
import risk_engine
import signals as signal_lib
import triggers as trigger_lib
import validation
from agents.vendor_qualification.agent import VendorQualificationAgent

from tests.fixtures import complete_form, without


def _agent() -> VendorQualificationAgent:
    """An agent with no Vertex/AI initialisation — these paths are pure Python."""
    return object.__new__(VendorQualificationAgent)


def _dossier(form, narrative=None):
    form = form or complete_form()
    bag = signal_lib.build_signals(form)
    assessment = risk_engine.assess(bag)
    override = trigger_lib.apply_mandatory_edd(assessment, bag)
    return _agent()._assemble(
        form=form,
        signal_bag=bag,
        retrieval={"jurisdiction_tier": "low", "jurisdiction_basis": ""},
        narrative=narrative or {},
        assessment=assessment,
        override=override,
        citation_list=[],
        retrieval_available=False,
    )


def _passing_narrative() -> dict:
    return {
        "company_profile": {
            "legal_entity_name": "Apex Gulf Technical Solutions LLC",
            "trade_license_no": "CN-1094821",
            "country_of_incorporation": "United Arab Emirates",
            "year_established": 2015,
            "business_category": "Strategic Supplier",
            "vat_registration_no": "100293847500003",
            "years_operating": 11,
        },
        "ownership_summary": {
            "structure_complexity": "SIMPLE",
            "structure_narrative": "Two UAE individuals hold the shares directly.",
            "pep_present": False,
            "ubos": [
                {"name": "Rashid Al-Farsi", "nationality": "United Arab Emirates",
                 "ownership_percentage": 60},
            ],
        },
        "appendix_f_assessment": {
            "financial_standing": 90,
            "technical_capability": 85,
            "quality_hse": 80,
            "rationale": "Turnover is 7x the proposed spend with three years of growth.",
            "strengths": ["Consistent turnover"],
            "concerns": [],
        },
        "required_controls": [
            {"control": "Annual financial monitoring", "basis": "Vendor Category Risk Treatment", "mandatory": True},
        ],
        "open_questions": [],
        "assessment_narrative": "A well-established engineering supplier with a growing order book.",
        "rules": [],
    }


class ReferenceFormTest(unittest.TestCase):
    def test_a_clearly_flagged_vendor_audits_clean(self):
        # The reference submission has a strategic AED 2.5M spend, so it must
        # come back NEEDS_EDD / High Risk at 2.50 or better.
        dossier = _dossier(complete_form(), _passing_narrative())
        self.assertTrue(dossier["mandatory_edd_triggered"])
        self.assertEqual(dossier["assigned_risk_tier"], "High Risk (EDD)")
        self.assertEqual(dossier["qualification_status"], "NEEDS_EDD")
        self.assertGreaterEqual(dossier["weighted_risk_score"], 2.5)

        audit = contract.audit(dossier)
        self.assertEqual(audit["invariant_violations"], [])
        self.assertTrue(audit["valid"], msg=str(audit["violations"]))

    def test_the_audit_never_removes_the_schema_skip_notice(self):
        # Whether or not jsonschema is installed, the caller is told.
        audit = contract.audit(_dossier(complete_form(), _passing_narrative()))
        if not contract.json_schema_available():
            self.assertFalse(audit["schema_checked"])
        else:
            self.assertTrue(audit["schema_checked"])


class TamperTest(unittest.TestCase):
    def _assert_caught(self, dossier, needle):
        audit = contract.audit(dossier)
        self.assertFalse(audit["valid"], msg="tampering was not detected")
        self.assertTrue(
            any(needle in v for v in audit["violations"]),
            msg=f"{needle!r} not in {audit['violations']}",
        )
        self.assertFalse(dossier["guardrail_check_passed"])

    def test_an_inflated_score_breaks_the_band_relationship(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        dossier["weighted_risk_score"] = 1.05
        dossier["assigned_risk_tier"] = "High Risk (EDD)"
        self._assert_caught(dossier, "policy band table")

    def test_a_score_outside_the_policy_range_is_caught(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        dossier["weighted_risk_score"] = 4.2
        self._assert_caught(dossier, "outside the policy range")

    def test_a_tier_that_is_not_a_policy_label_is_caught(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        dossier["assigned_risk_tier"] = "Very High Risk"
        self._assert_caught(dossier, "is not one of")

    def test_an_edd_trigger_with_a_low_tier_is_caught(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        dossier["assigned_risk_tier"] = "Medium Risk (CDD)"
        self._assert_caught(dossier, "the policy requires")

    def test_an_edd_trigger_below_the_floor_is_caught(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        dossier["weighted_risk_score"] = 2.20
        self._assert_caught(dossier, "at least")

    def test_an_edd_trigger_reported_as_qualified_is_caught(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        dossier["qualification_status"] = "QUALIFIED"
        self._assert_caught(dossier, "cannot produce a QUALIFIED status")

    def test_trigger_reasons_disagreeing_with_the_flag_is_caught(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        dossier["trigger_reasons"] = []
        self._assert_caught(dossier, "trigger_reasons holds")

    def test_appendix_f_components_that_do_not_sum_are_caught(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        dossier["appendix_f_score"]["financial_standing_score"] = 1.0
        self._assert_caught(dossier, "add up to")

    def test_an_appendix_f_passed_below_the_threshold_is_caught(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        appendix = dossier["appendix_f_score"]
        # The sheet is tier-derived (FAILED at 55.0 on this EDD fixture), so
        # the tamper marks it PASSED as well — the invariant must then catch
        # a pass claimed below the threshold.
        appendix["status"] = "PASSED"
        appendix["total_score"] = 20.0
        appendix["financial_standing_score"] = 7.0
        appendix["technical_capability_score"] = 7.0
        appendix["quality_hse_score"] = 6.0
        self._assert_caught(dossier, "marked PASSED but totals")

    def test_an_incomplete_verdict_carrying_a_score_is_caught(self):
        incomplete = without(complete_form(), "trade_license_no")
        dossier = _agent()._incomplete_dossier(incomplete, validation.validate(incomplete))
        self.assertIsNone(dossier["weighted_risk_score"])
        self.assertTrue(contract.audit(dossier)["valid"])

        dossier["weighted_risk_score"] = 1.0
        self._assert_caught(dossier, "no assessment was performed")

    def test_a_verified_citation_without_evidence_is_caught(self):
        dossier = _dossier(complete_form(), _passing_narrative())
        dossier["rag_retrieval_citations"] = [
            {
                "document": "Procurement Policy_Updated_2.pdf",
                "section": "Risk Factors",
                "rule_applied": "Vendors in high-risk jurisdictions are High Risk.",
                "quote": "",
                "source_uri": "",
                "score": 0.0,
                "verified": True,
                "matched_trigger": None,
                "factor": "jurisdiction",
            }
        ]
        self._assert_caught(dossier, "carries no quote")


class HonestEmptyVendorTest(unittest.TestCase):
    def test_an_empty_submission_is_incomplete_with_no_score(self):
        dossier = _agent()._incomplete_dossier({}, validation.validate({}))
        self.assertEqual(dossier["qualification_status"], "REJECTED_INCOMPLETE")
        self.assertIsNone(dossier["weighted_risk_score"])
        self.assertIsNone(dossier["assigned_risk_tier"])
        self.assertFalse(dossier["mandatory_edd_triggered"])
        self.assertFalse(dossier["guardrail_check_passed"])
        self.assertTrue(dossier["missing_data_request"]["fields"])

        audit = contract.audit(dossier)
        self.assertTrue(audit["valid"], msg=str(audit["violations"]))


class TriggerReasonTest(unittest.TestCase):
    def test_flat_reasons_match_the_audit_expectation(self):
        bag = signal_lib.build_signals(complete_form())
        override = trigger_lib.apply_mandatory_edd(risk_engine.assess(bag), bag)
        self.assertEqual(
            len(trigger_lib.trigger_reasons(override.triggers)),
            len(override.triggers),
        )
        self.assertEqual(
            trigger_lib.trigger_reasons(override.triggers)[0],
            override.triggers[0].label,
        )
        self.assertIsInstance(override.score_override, Decimal)


if __name__ == "__main__":
    unittest.main()
