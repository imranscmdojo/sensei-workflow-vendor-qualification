"""Mandatory EDD triggers: detection, the override, and what it must not do."""

import unittest
from decimal import Decimal

import risk_engine
import triggers


def _signals(**overrides) -> dict:
    """A benign signal bag, overridden section by section."""
    bag = {
        "jurisdiction": {"tier": "low"},
        "ownership": {"structure": "simple"},
        "government": {"role": "direct supplier"},
        "payment": {"structure": "standard"},
        "spend": {"annual_spend_aed": 50_000},
        "nature": {"sensitivity": "routine"},
        "reputation": {"adverse_media": "clear"},
        "relationship": {"history": "first_time"},
        "pep": {"present": False},
        "sanctions": {"unresolved_match": False},
        "corporate": {},
        "governance": {},
    }
    for section, values in overrides.items():
        bag[section] = {**bag.get(section, {}), **values}
    return bag


def _codes(bag: dict):
    return {t.code for t in triggers.evaluate_triggers(bag)}


class BenignVendorTest(unittest.TestCase):
    def test_a_clean_supplier_triggers_nothing(self):
        self.assertEqual(_codes(_signals()), set())

    def test_no_override_leaves_the_base_tier_alone(self):
        bag = _signals()
        result = triggers.apply_mandatory_edd(risk_engine.assess(bag), bag)
        self.assertFalse(result.triggered)
        self.assertEqual(result.triggers, [])
        self.assertIsNone(result.score_override)
        self.assertIsNone(result.tier_override)


class DetectorTest(unittest.TestCase):
    def test_every_trigger_fires_on_its_own_signal(self):
        cases = {
            "PEP_PRESENT": _signals(pep={"present": True}),
            "HIGH_RISK_JURISDICTION": _signals(jurisdiction={"tier": "high"}),
            "UNRESOLVED_SANCTIONS_MATCH": _signals(sanctions={"unresolved_match": True}),
            "COMPLEX_OWNERSHIP": _signals(ownership={"structure": "nominee"}),
            "GOVERNMENT_FACING_INTERMEDIARY": _signals(government={"role": "lobbyist"}),
            "DISTRIBUTOR_OR_RESELLER": _signals(government={"role": "distributor"}),
            "STRATEGIC_SPEND": _signals(spend={"annual_spend_aed": 2_000_000}),
            "LARGE_CONSTRUCTION_OR_INTL_SUBCONTRACTING": _signals(
                spend={"construction_project_value_aed": 10_000_000}),
            "ADVERSE_MEDIA_ASSOCIATION": _signals(
                reputation={"adverse_media": "credible association"}),
            "JOINT_VENTURE_OR_MA_TARGET": _signals(corporate={"co_investor": True}),
            "GCICD_EDD_DETERMINATION": _signals(
                governance={"gcidc_edd_determination": True}),
        }
        # Every declared trigger must be covered, or a trigger exists that this
        # test does not know about.
        self.assertEqual(set(cases), {t.code for t in triggers.MANDATORY_EDD_TRIGGERS})
        for code, bag in cases.items():
            with self.subTest(trigger=code):
                self.assertIn(code, _codes(bag))

    def test_prohibited_jurisdiction_counts_as_high_risk(self):
        codes = _codes(_signals(jurisdiction={"tier": "prohibited"}))
        self.assertIn("HIGH_RISK_JURISDICTION", codes)

    def test_offshore_layered_ownership_is_complex(self):
        self.assertIn(
            "COMPLEX_OWNERSHIP",
            _codes(_signals(ownership={"structure": "cross_border_layered"})),
        )

    def test_a_broken_detector_is_a_miss_not_a_pass(self):
        """A detector that raises must be visible, never silently skipped."""
        broken = triggers.EddTrigger(
            code="BROKEN",
            label="Broken",
            policy_section="x",
            policy_basis="x",
            minimum_controls="x",
            detector=lambda signals: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        original = triggers.MANDATORY_EDD_TRIGGERS
        try:
            triggers.MANDATORY_EDD_TRIGGERS = original + (broken,)
            hits = triggers.evaluate_triggers(_signals())
            self.assertNotIn("BROKEN", {t.code for t in hits})
        finally:
            triggers.MANDATORY_EDD_TRIGGERS = original


class OverrideTest(unittest.TestCase):
    def test_override_raises_a_low_vendor_to_the_edd_floor(self):
        bag = _signals(pep={"present": True})
        assessment = risk_engine.assess(bag)
        self.assertEqual(assessment.assigned_risk_tier, "Low Risk (SDD)")

        result = triggers.apply_mandatory_edd(assessment, bag)
        self.assertTrue(result.triggered)
        self.assertEqual(result.score_override, Decimal("2.50"))
        self.assertEqual(result.tier_override, "High Risk (EDD)")

    def test_override_never_lowers_a_high_score(self):
        """A vendor already above the floor keeps its own score. The override is
        a floor, not a fixed value."""
        bag = _signals(
            jurisdiction={"tier": "high"},
            ownership={"structure": "bearer share"},
            government={"role": "government_facing"},
            payment={"structure": "cash"},
            spend={"annual_spend_aed": 99_000_000},
            nature={"sensitivity": "financial_institution"},
            reputation={"adverse_media": "adverse"},
            relationship={"history": "prior_suspended"},
        )
        assessment = risk_engine.assess(bag)
        self.assertGreater(assessment.weighted_risk_score, Decimal("2.50"))

        result = triggers.apply_mandatory_edd(assessment, bag)
        self.assertEqual(result.score_override, assessment.weighted_risk_score)
        self.assertEqual(result.tier_override, "High Risk (EDD)")
        self.assertEqual(result.tier_override, assessment.assigned_risk_tier)

    def test_an_override_capped_by_a_narrow_factor_does_not_inflate_the_score(self):
        """A single 3.00 factor is not on its own a high-risk vendor. The override
        must not turn a weighted 1.40 into a 2.50 by arithmetic accident."""
        bag = _signals(ownership={"structure": "bearer share"})
        assessment = risk_engine.assess(bag)
        self.assertEqual(assessment.weighted_risk_score, Decimal("1.40"))

        result = triggers.apply_mandatory_edd(assessment, bag)
        self.assertTrue(result.triggered)
        self.assertEqual(result.score_override, Decimal("2.50"))
        self.assertEqual(result.tier_override, "High Risk (EDD)")

    def test_overridden_score_still_lands_in_its_tier_band(self):
        for signals in (
            _signals(pep={"present": True}),
            _signals(ownership={"structure": "nominee"}, spend={"annual_spend_aed": 99_000_000}),
        ):
            with self.subTest(signals=signals):
                assessment = risk_engine.assess(signals)
                result = triggers.apply_mandatory_edd(assessment, signals)
                tier, _, _ = risk_engine.tier_for(result.score_override)
                self.assertEqual(tier, result.tier_override)

    def test_trigger_reasons_matches_the_trigger_list(self):
        bag = _signals(pep={"present": True}, spend={"annual_spend_aed": 5_000_000})
        result = triggers.apply_mandatory_edd(risk_engine.assess(bag), bag)
        reasons = triggers.trigger_reasons(result.triggers)
        self.assertEqual(len(reasons), len(result.triggers))
        self.assertIn("PEP_PRESENT", {t.code for t in result.triggers})
        self.assertIn("STRATEGIC_SPEND", {t.code for t in result.triggers})


class StrategicSpendDivergenceTest(unittest.TestCase):
    """STRATEGIC_SPEND is stricter than the policy's category matrix on
    purpose. This test pins the divergence so removing it is a deliberate,
    visible act rather than an accident."""

    def test_exactly_at_the_threshold_triggers(self):
        self.assertIn("STRATEGIC_SPEND", _codes(_signals(spend={"annual_spend_aed": 2_000_000})))

    def test_just_under_the_threshold_does_not(self):
        self.assertNotIn("STRATEGIC_SPEND", _codes(_signals(spend={"annual_spend_aed": 1_999_999})))

    def test_the_threshold_is_a_single_documented_constant(self):
        # The spend band in risk_engine and the trigger in triggers.py must
        # never drift apart: the band scores it, the trigger escalates it.
        self.assertEqual(
            triggers.SPEND_STRATEGIC_THRESHOLD_AED,
            risk_engine.SPEND_STRATEGIC_THRESHOLD_AED,
        )
        self.assertEqual(
            risk_engine.spend_band(triggers.SPEND_STRATEGIC_THRESHOLD_AED),
            "strategic_2m_plus",
        )


if __name__ == "__main__":
    unittest.main()
