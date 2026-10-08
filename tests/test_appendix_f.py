"""Appendix F score sheet: the 35/35/30 weighting, the total, and the gate."""

import unittest

from agents.vendor_qualification.schemas import (
    APPENDIX_F_PASS_THRESHOLD,
    APPENDIX_F_WEIGHTS,
    appendix_f_scores,
)


def _raw(financial: int, technical: int, quality: int) -> dict:
    return {
        "financial_standing": financial,
        "technical_capability": technical,
        "quality_hse": quality,
        "rationale": "test",
        "strengths": [],
        "concerns": [],
    }


class WeightingTest(unittest.TestCase):
    def test_weights_sum_to_one_hundred(self):
        self.assertAlmostEqual(sum(APPENDIX_F_WEIGHTS.values()), 100.0, places=9)

    def test_a_perfect_sheet_is_one_hundred(self):
        result = appendix_f_scores(_raw(100, 100, 100))
        self.assertEqual(result["total_score"], 100.0)
        self.assertEqual(result["financial_standing_score"], 35.0)
        self.assertEqual(result["technical_capability_score"], 35.0)
        self.assertEqual(result["quality_hse_score"], 30.0)
        self.assertEqual(result["status"], "PASSED")

    def test_the_worked_example_from_output_schema_md(self):
        """output_schema.md shows 35.0 + 29.5 + 20.0 = 84.5, which fixes the
        sub-weights as 35 / 35 / 30."""
        result = appendix_f_scores(_raw(100, 84.3, 66.7))
        self.assertEqual(result["financial_standing_score"], 35.0)
        self.assertEqual(result["technical_capability_score"], 29.5)
        self.assertEqual(result["quality_hse_score"], 20.0)
        self.assertEqual(result["total_score"], 84.5)
        self.assertEqual(result["status"], "PASSED")

    def test_an_empty_assessment_is_zero_not_a_pass(self):
        result = appendix_f_scores({})
        self.assertEqual(result["total_score"], 0.0)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(
            set(result["failed_categories"]),
            {"financial_standing", "technical_capability", "quality_hse"},
        )


class GateTest(unittest.TestCase):
    def test_exactly_at_the_threshold_passes(self):
        # 24.5 + 24.5 + 21.0 = 70.0
        result = appendix_f_scores(_raw(70, 70, 70))
        self.assertEqual(result["total_score"], APPENDIX_F_PASS_THRESHOLD)
        self.assertEqual(result["status"], "PASSED")

    def test_just_under_the_threshold_fails(self):
        result = appendix_f_scores(_raw(69, 69, 69))
        self.assertLess(result["total_score"], APPENDIX_F_PASS_THRESHOLD)
        self.assertEqual(result["status"], "FAILED")

    def test_a_strong_technical_score_cannot_carry_a_weak_financial_one(self):
        """Total clears 70 but financial is below half of its 35 maximum."""
        result = appendix_f_scores(_raw(30, 100, 100))
        self.assertGreaterEqual(result["total_score"], APPENDIX_F_PASS_THRESHOLD)
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("financial_standing", result["failed_categories"])
        self.assertNotIn("technical_capability", result["failed_categories"])

    def test_the_category_minimum_is_half_of_each_maximum(self):
        # 17.5 of 35 is the financial minimum; 17.49 of it fails.
        self.assertEqual(
            appendix_f_scores(_raw(50, 100, 100))["status"], "PASSED"
        )
        self.assertEqual(
            appendix_f_scores(_raw(49, 100, 100))["status"], "FAILED"
        )


class ClampingTest(unittest.TestCase):
    def test_out_of_range_values_are_clamped_not_rejected(self):
        high = appendix_f_scores(_raw(150, 150, 150))
        self.assertEqual(high["total_score"], 100.0)
        low = appendix_f_scores(_raw(-20, -5, -1))
        self.assertEqual(low["total_score"], 0.0)

    def test_non_numeric_values_score_zero(self):
        result = appendix_f_scores(_raw("excellent", None, []))
        self.assertEqual(result["total_score"], 0.0)
        self.assertEqual(result["status"], "FAILED")

    def test_raw_assessments_are_preserved_for_review(self):
        result = appendix_f_scores(_raw(80, 60, 55))
        self.assertEqual(
            result["raw_assessments"],
            {"financial_standing": 80.0, "technical_capability": 60.0, "quality_hse": 55.0},
        )


class RationaleTest(unittest.TestCase):
    def test_rationale_strengths_and_concerns_survive(self):
        result = appendix_f_scores({
            **_raw(80, 60, 55),
            "rationale": "  Turnover is 7x the proposed spend.  ",
            "strengths": ["Stable turnover"],
            "concerns": ["No ISO 45001"],
        })
        self.assertEqual(result["rationale"], "Turnover is 7x the proposed spend.")
        self.assertEqual(result["strengths"], ["Stable turnover"])
        self.assertEqual(result["concerns"], ["No ISO 45001"])


if __name__ == "__main__":
    unittest.main()
