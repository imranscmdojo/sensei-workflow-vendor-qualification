"""Deterministic weighted risk score engine: bands, rounding, clamping, aliases."""

import unittest
from decimal import Decimal

import risk_engine


def _all_signals_at(value: str) -> dict:
    """A signal bag with every factor pinned to the same normalised value."""
    return {
        "jurisdiction": {"tier": value},
        "ownership": {"structure": value},
        "government": {"role": value},
        "payment": {"structure": value},
        "spend": {"band": value},
        "nature": {"sensitivity": value},
        "reputation": {"adverse_media": value},
        "relationship": {"history": value},
    }


class WeightTableTest(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(risk_engine.FACTOR_WEIGHTS.values()), 1.0, places=9)

    def test_every_weighted_factor_has_a_label_and_rules(self):
        for factor, weight in risk_engine.FACTOR_WEIGHTS.items():
            self.assertIn(factor, risk_engine.FACTOR_LABELS)
            self.assertGreater(weight, 0.0)
        # factor_catalog is what the UI renders; it must cover all eight.
        self.assertEqual(len(risk_engine.factor_catalog()), 8)


class TierBandTest(unittest.TestCase):
    def test_band_edges_are_inclusive_upper_and_verbatim(self):
        cases = [
            (Decimal("1.00"), "Low Risk (SDD)", "Simplified Due Diligence (SDD)", "Every 3 Years"),
            (Decimal("1.60"), "Low Risk (SDD)", "Simplified Due Diligence (SDD)", "Every 3 Years"),
            (Decimal("1.61"), "Medium Risk (CDD)", "Standard Customer Due Diligence (CDD)", "Every 2 Years"),
            (Decimal("2.20"), "Medium Risk (CDD)", "Standard Customer Due Diligence (CDD)", "Every 2 Years"),
            (Decimal("2.21"), "High Risk (EDD)", "Enhanced Due Diligence (EDD)", "Annual Refresh"),
            (Decimal("3.00"), "High Risk (EDD)", "Enhanced Due Diligence (EDD)", "Annual Refresh"),
        ]
        for score, tier, dd_level, cycle in cases:
            with self.subTest(score=score):
                self.assertEqual(
                    risk_engine.tier_for(score), (tier, dd_level, cycle)
                )

    def test_scores_below_the_floor_still_map(self):
        self.assertEqual(
            risk_engine.tier_for(Decimal("0.50"))[0], "Low Risk (SDD)"
        )


class RoundingTest(unittest.TestCase):
    def test_rounds_half_up_not_bankers(self):
        """2.205 must become 2.21. Python's default rounding would give 2.20
        and drop the score below a band edge."""
        from risk_engine import FactorResult

        factors = [
            FactorResult("annual_spend", "Spend", 1.0, Decimal("2.205"), "r", "b"),
        ]
        self.assertEqual(risk_engine.weighted_score(factors), Decimal("2.21"))

    def test_clamps_to_policy_range(self):
        from risk_engine import FactorResult

        floor = [FactorResult("f", "F", 1.0, Decimal("0.10"), "r", "b")]
        ceiling = [FactorResult("f", "F", 1.0, Decimal("9.99"), "r", "b")]
        self.assertEqual(risk_engine.weighted_score(floor), Decimal("1.00"))
        self.assertEqual(risk_engine.weighted_score(ceiling), Decimal("3.00"))

    def test_weights_are_actually_applied(self):
        """A single factor at weight 0.25 moves the total by a quarter of its
        delta, which catches a sum-of-scores implementation."""
        from risk_engine import FactorResult

        factors = [
            FactorResult("jurisdiction", "J", 0.25, Decimal("3.00"), "r", "b"),
            FactorResult("ownership_control", "O", 0.20, Decimal("1.00"), "r", "b"),
            FactorResult("government_exposure", "G", 0.15, Decimal("1.00"), "r", "b"),
            FactorResult("payment_structure", "P", 0.15, Decimal("1.00"), "r", "b"),
            FactorResult("annual_spend", "S", 0.10, Decimal("1.00"), "r", "b"),
            FactorResult("nature_sensitivity", "N", 0.05, Decimal("1.00"), "r", "b"),
            FactorResult("adverse_media", "A", 0.05, Decimal("1.00"), "r", "b"),
            FactorResult("prior_relationship", "R", 0.05, Decimal("1.00"), "r", "b"),
        ]
        # 0.25*3 + 0.75*1 = 1.50
        self.assertEqual(risk_engine.weighted_score(factors), Decimal("1.50"))


class UnknownSignalTest(unittest.TestCase):
    def test_absent_factor_is_unknown_not_clean(self):
        """A factor with no signal must not score as the 1.00 floor."""
        factors = risk_engine.score_factors({})
        for factor in factors:
            with self.subTest(factor=factor.factor):
                self.assertIsNone(factor.signal)
                self.assertEqual(factor.score, risk_engine.SCORE_UNKNOWN)

    def test_unrecognised_signal_falls_back_to_unknown(self):
        factors = risk_engine.score_factors(
            {"jurisdiction": {"tier": "atlantis"}}
        )
        jurisdiction = next(f for f in factors if f.factor == "jurisdiction")
        self.assertEqual(jurisdiction.score, risk_engine.SCORE_UNKNOWN)

    def test_undeclared_reputation_is_unknown_not_clear(self):
        """Stage 3 screening has not run, so reputational risk is unassessed."""
        factors = risk_engine.score_factors({"reputation": {}})
        reputation = next(f for f in factors if f.factor == "adverse_media")
        self.assertEqual(reputation.score, risk_engine.SCORE_UNKNOWN)


class AliasTest(unittest.TestCase):
    def test_prose_ownership_maps_to_the_same_rule(self):
        prose = {"ownership": {"structure": "Multi-layered offshore holding structure"}}
        factors = risk_engine.score_factors(prose)
        ownership = next(f for f in factors if f.factor == "ownership_control")
        self.assertEqual(ownership.signal, "layered")
        self.assertEqual(ownership.score, Decimal("2.00"))

    def test_boolean_flags_fall_back_to_rules(self):
        factors = risk_engine.score_factors(
            {"ownership": {"nominee": True}, "payment": {"cash": True}}
        )
        by_name = {f.factor: f for f in factors}
        self.assertEqual(by_name["ownership_control"].score, Decimal("3.00"))
        self.assertEqual(by_name["payment_structure"].score, Decimal("3.00"))


class SpendBandTest(unittest.TestCase):
    def test_bands_respect_the_policy_thresholds(self):
        cases = [
            (50_000, "under_100k"),
            (99_999, "under_100k"),
            (100_000, "under_500k"),
            (499_999, "under_500k"),
            (500_000, "under_2m"),
            (1_999_999, "under_2m"),
            (2_000_000, "strategic_2m_plus"),
            (9_999_999, "strategic_2m_plus"),
            (10_000_000, "strategic_10m_plus"),
        ]
        for amount, expected in cases:
            with self.subTest(amount=amount):
                self.assertEqual(risk_engine.spend_band(amount), expected)

    def test_single_contract_value_can_set_the_band(self):
        self.assertEqual(
            risk_engine.spend_band(10_000, 3_000_000), "strategic_2m_plus"
        )

    def test_no_spend_declared_is_unknown(self):
        self.assertEqual(risk_engine.spend_band(None), "unknown")


class AssessmentTest(unittest.TestCase):
    def test_assess_reports_unknown_factors(self):
        assessment = risk_engine.assess({})
        self.assertEqual(len(assessment.factors), 8)
        self.assertEqual(len(assessment.unknown_factors), 8)
        # Everything unknown sits at 2.00, so the total is 2.00.
        self.assertEqual(assessment.weighted_risk_score, Decimal("2.00"))
        self.assertEqual(assessment.assigned_risk_tier, "Medium Risk (CDD)")

    def test_assess_is_deterministic(self):
        signals = _all_signals_at("simple")
        first = risk_engine.assess(signals)
        second = risk_engine.assess(signals)
        self.assertEqual(first.weighted_risk_score, second.weighted_risk_score)
        self.assertEqual(first.assigned_risk_tier, second.assigned_risk_tier)

    def test_best_case_vendor_is_low_risk_at_the_floor(self):
        signals = {
            "jurisdiction": {"tier": "low"},
            "ownership": {"structure": "simple"},
            "government": {"role": "direct supplier"},
            "payment": {"structure": "standard"},
            "spend": {"annual_spend_aed": 50_000},
            "nature": {"sensitivity": "routine"},
            "reputation": {"adverse_media": "clear"},
            "relationship": {"history": "prior_qualified"},
        }
        assessment = risk_engine.assess(signals)
        self.assertEqual(assessment.weighted_risk_score, Decimal("1.00"))
        self.assertEqual(assessment.assigned_risk_tier, "Low Risk (SDD)")
        self.assertEqual(assessment.unknown_factors, [])

    def test_worst_case_vendor_is_ceilinged(self):
        signals = {
            "jurisdiction": {"tier": "prohibited"},
            "ownership": {"structure": "bearer share"},
            "government": {"role": "lobbyist"},
            "payment": {"structure": "cash"},
            "spend": {"annual_spend_aed": 50_000_000},
            "nature": {"sensitivity": "critical"},
            "reputation": {"adverse_media": "credible association"},
            "relationship": {"history": "blacklisted"},
        }
        assessment = risk_engine.assess(signals)
        self.assertEqual(assessment.weighted_risk_score, Decimal("3.00"))
        self.assertEqual(assessment.assigned_risk_tier, "High Risk (EDD)")


if __name__ == "__main__":
    unittest.main()
