"""
Agent orchestration: the Zero-Hallucination gate, the RAG fallback, the SSE
event sequence, and the fact that the score is never the model's to choose.

No network and no model. The two generation methods are replaced on the
instance, so these tests assert on the wiring around them.
"""

import asyncio
import unittest

import contract
import risk_engine
import signals as signal_lib
import triggers as trigger_lib
from agents.base_rag_agent import LLMParseError
from agents.vendor_qualification import schemas as vq_schemas
from agents.vendor_qualification.agent import VendorQualificationAgent, _vendor_id

from tests.fixtures import complete_form, without

POLICY_CHUNK = {
    "uri": "https://drive.google.com/file/d/1AbC/view",
    "text": (
        "Mandatory High-Risk Classification "
        "Regardless of the weighted score, the following Vendors are always "
        "classified as High Risk and are subject to Enhanced Due Diligence: "
        "Any Vendor domiciled, incorporated, operating, or beneficially owned "
        "in a High-Risk Jurisdiction. Any Vendor where a positive or potential "
        "sanctions match has arisen and has not been conclusively resolved as a "
        "false positive by the MLRO."
    ),
    "score": 0.71,
}

RETRIEVAL_RESULT = {
    "data": {
        "jurisdiction_tier": "low",
        "jurisdiction_basis": "The Country Risk List tier for the country.",
        "business_category": "Strategic Supplier",
        "rules": [
            {
                "section": "Mandatory High-Risk Classification",
                "rule_applied": (
                    "A sanctions match that the MLRO has not conclusively "
                    "resolved blocks onboarding."
                ),
                "matched_trigger": "UNRESOLVED_SANCTIONS_MATCH",
                "factor": "",
            }
        ],
    },
    "chunks": [POLICY_CHUNK],
}

NARRATIVE_RESULT = {
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
        "ubos": [],
    },
    "appendix_f_assessment": {
        "financial_standing": 90,
        "technical_capability": 85,
        "quality_hse": 80,
        "rationale": "Turnover is 7x the proposed spend.",
        "strengths": [],
        "concerns": [],
    },
    "required_controls": [],
    "open_questions": [],
    "assessment_narrative": "An established engineering supplier.",
    "rules": [],
}


def _events(agent, form):
    async def run():
        return [event async for event in agent.qualify_stream(form)]
    return asyncio.run(run())


def _dossier_from(events):
    dossiers = [e["dossier"] for e in events if e["type"] == "dossier"]
    assert dossiers, "the stream must end with a dossier"
    return dossiers[-1]


def _raise_on_narrative(**kwargs):
    """Fails the Phase B narrative while leaving Phase A retrieval working."""
    if kwargs.get("step") == "phase_b_narrative":
        raise LLMParseError("truncated json")
    return {"data": {"verbatim_text": []}, "chunks": [POLICY_CHUNK]}


class ZeroHallucinationGateTest(unittest.TestCase):
    """The most important test in the suite: an incomplete submission must not
    reach a model at all."""

    def _forbid_the_model(self, agent):
        def forbidden(*args, **kwargs):
            raise AssertionError("the model was called on an incomplete submission")
        agent.generate_grounded = forbidden
        agent.generate_with_rag = forbidden

    def test_an_incomplete_submission_never_calls_the_model(self):
        agent = VendorQualificationAgent()
        self._forbid_the_model(agent)

        events = _events(agent, without(complete_form(), "trade_license_no"))
        dossier = _dossier_from(events)

        self.assertEqual(dossier["qualification_status"], "REJECTED_INCOMPLETE")
        self.assertIsNone(dossier["weighted_risk_score"])
        self.assertIsNone(dossier["assigned_risk_tier"])
        self.assertFalse(dossier["mandatory_edd_triggered"])
        self.assertEqual(dossier["rag_retrieval_citations"], [])
        self.assertEqual(
            [m["field"] for m in dossier["missing_data_request"]["fields"]],
            ["trade_license_no"],
        )

    def test_an_empty_submission_reports_every_gap_and_no_score(self):
        agent = VendorQualificationAgent()
        self._forbid_the_model(agent)

        dossier = _dossier_from(_events(agent, {}))
        self.assertEqual(dossier["qualification_status"], "REJECTED_INCOMPLETE")
        self.assertIsNone(dossier["weighted_risk_score"])
        self.assertGreater(dossier["missing_data_request"]["count"], 20)
        self.assertIn("NH-PQF-001", dossier["missing_data_request"]["message"])

    def test_no_score_event_is_emitted_for_an_incomplete_submission(self):
        agent = VendorQualificationAgent()
        self._forbid_the_model(agent)
        kinds = [e["type"] for e in _events(agent, {})]
        self.assertNotIn("score", kinds)
        self.assertNotIn("triggers", kinds)
        self.assertEqual(kinds, ["stage", "validation", "stage", "dossier"])

    def test_qualify_short_circuits_too(self):
        agent = VendorQualificationAgent()
        self._forbid_the_model(agent)
        dossier = asyncio.run(agent.qualify(without(complete_form(), "ubos")))
        self.assertEqual(dossier["qualification_status"], "REJECTED_INCOMPLETE")


class FullRunTest(unittest.TestCase):
    def _stubbed_agent(self, retrieval=RETRIEVAL_RESULT, narrative=NARRATIVE_RESULT):
        agent = VendorQualificationAgent()
        calls = {"retrieval": 0, "narrative": 0}

        def fake_grounded(**kwargs):
            calls["retrieval"] += 1
            return retrieval

        async def fake_narrative(**kwargs):
            calls["narrative"] += 1
            if isinstance(narrative, Exception):
                raise narrative
            return narrative

        agent.generate_grounded = fake_grounded
        agent.generate_with_rag = fake_narrative
        return agent, calls

    def test_the_dossier_reports_the_engine_s_numbers(self):
        agent, _ = self._stubbed_agent()
        dossier = _dossier_from(_events(agent, complete_form()))

        # 2.5M spend is a mandatory EDD trigger, so High Risk at >= 2.50.
        self.assertTrue(dossier["mandatory_edd_triggered"])
        self.assertEqual(dossier["assigned_risk_tier"], "High Risk (EDD)")
        self.assertGreaterEqual(dossier["weighted_risk_score"], 2.50)
        self.assertEqual(dossier["qualification_status"], "NEEDS_EDD")
        self.assertIn("STRATEGIC_SPEND", {t["code"] for t in dossier["trigger_details"]})
        self.assertTrue(dossier["guardrail_check_passed"])
        self.assertTrue(contract.audit(dossier)["valid"])

    def test_the_retrieved_country_tier_feeds_the_engine(self):
        agent, _ = self._stubbed_agent()
        dossier = _dossier_from(_events(agent, complete_form()))
        self.assertEqual(dossier["jurisdiction_assessment"]["tier_source"], "rag")
        self.assertEqual(dossier["jurisdiction_assessment"]["tier"], "low")
        # The rag tier is the lowest jurisdiction anchor in the table.
        jurisdiction = next(
            f for f in dossier["weighted_factors"] if f["factor"] == "jurisdiction"
        )
        self.assertEqual(jurisdiction["score"], 1.0)

    def test_citations_are_backed_by_retrieved_text(self):
        agent, _ = self._stubbed_agent()
        dossier = _dossier_from(_events(agent, complete_form()))

        citations = dossier["rag_retrieval_citations"]
        self.assertEqual(len(citations), 1)
        citation = citations[0]
        self.assertTrue(citation["verified"])
        self.assertIn("Mandatory High-Risk Classification", citation["quote"])
        self.assertEqual(citation["source_uri"], POLICY_CHUNK["uri"])
        self.assertEqual(citation["matched_trigger"], "UNRESOLVED_SANCTIONS_MATCH")
        self.assertEqual(dossier["citation_summary"]["verified"], 1)

    def test_an_unverifiable_assertion_is_kept_but_labelled(self):
        retrieval = {
            "data": {
                "jurisdiction_tier": "undetermined",
                "jurisdiction_basis": "",
                "business_category": "Undetermined",
                "rules": [
                    {
                        "section": "A Section That Is Not In The Corpus",
                        "rule_applied": "Totally unrelated assertion about widgets.",
                        "matched_trigger": "",
                        "factor": "",
                    }
                ],
            },
            "chunks": [POLICY_CHUNK],
        }
        agent, _ = self._stubbed_agent(retrieval=retrieval)
        dossier = _dossier_from(_events(agent, complete_form()))

        citation = dossier["rag_retrieval_citations"][0]
        self.assertFalse(citation["verified"])
        self.assertEqual(citation["quote"], "")
        self.assertEqual(dossier["citation_summary"]["unverified"], 1)

    def test_a_verified_citation_and_an_unverified_one_are_both_reported(self):
        narrative = dict(NARRATIVE_RESULT)
        narrative["rules"] = [
            {
                "section": "Phase B Only Section",
                "rule_applied": "An assertion the corpus never mentions.",
                "matched_trigger": "",
                "factor": "",
            }
        ]
        agent, _ = self._stubbed_agent(narrative=narrative)
        dossier = _dossier_from(_events(agent, complete_form()))

        summary = dossier["citation_summary"]
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["verified"], 1)
        self.assertEqual(summary["unverified"], 1)
        unverified = [c for c in dossier["rag_retrieval_citations"]
                      if not c["verified"]]
        self.assertEqual(len(unverified), 1)
        self.assertEqual(unverified[0]["section"], "Phase B Only Section")

    def test_the_event_sequence_is_ordered_for_the_dashboard(self):
        agent, calls = self._stubbed_agent()
        events = _events(agent, complete_form())
        kinds = [e["type"] for e in events]

        # One vendor-specific retrieval plus one anchor query per policy section.
        self.assertEqual(calls["narrative"], 1)
        self.assertEqual(calls["retrieval"],
                         1 + len(vq_schemas.POLICY_ANCHOR_QUERIES))
        self.assertEqual(kinds[0], "stage")
        self.assertEqual(kinds[1], "validation")
        self.assertLess(kinds.index("retrieval"), kinds.index("score"))
        self.assertLess(kinds.index("score"), kinds.index("triggers"))
        self.assertLess(kinds.index("triggers"), kinds.index("dossier"))
        self.assertEqual(kinds[-1], "dossier")
        self.assertIn("jurisdiction", kinds)
        self.assertIn("category", kinds)

    def test_the_score_event_exposes_the_factor_table(self):
        agent, _ = self._stubbed_agent()
        score_events = [e for e in _events(agent, complete_form()) if e["type"] == "score"]
        self.assertEqual(len(score_events), 1)
        event = score_events[0]
        self.assertTrue(event["overridden"])
        self.assertEqual(len(event["factors"]), 8)
        # The pre-override score is still reported for the reviewer.
        self.assertLess(event["base_score"], event["weighted_risk_score"])
        self.assertEqual(event["base_tier"], "Low Risk (SDD)")

    def test_an_edd_vendor_is_never_told_to_go_straight_to_onboarding(self):
        """A mandatory EDD trigger means the vendor is not prequalified yet.
        Telling the officer to route it to the Vendor Onboarding Agent is the
        exact compliance error this tool exists to prevent, so the handover
        language is asserted on."""
        agent, _ = self._stubbed_agent()
        dossier = _dossier_from(_events(agent, complete_form()))

        self.assertEqual(dossier["qualification_status"], "NEEDS_EDD")
        action = dossier["next_action"]
        self.assertIn("Enhanced Due Diligence", action)
        self.assertIn("before any hand-off", action)

    def test_both_phases_are_requested_on_the_expected_models(self):
        agent = VendorQualificationAgent()
        seen = {"models": []}

        def fake_grounded(**kwargs):
            seen["models"].append(kwargs.get("model_name"))
            return RETRIEVAL_RESULT

        async def fake_narrative(**kwargs):
            seen["narrative_model"] = kwargs.get("model_name")
            seen["narrative_schema_has_no_score_fields"] = (
                "weighted_risk_score" not in kwargs["response_schema"]["properties"]
                and "assigned_risk_tier" not in kwargs["response_schema"]["properties"]
            )
            return NARRATIVE_RESULT

        agent.generate_grounded = fake_grounded
        agent.generate_with_rag = fake_narrative
        _events(agent, complete_form())

        # Every retrieval call, anchors included, is on the cheap model.
        self.assertTrue(seen["models"])
        self.assertEqual(set(seen["models"]), {"gemini-2.5-flash"})
        self.assertEqual(seen["narrative_model"], "gemini-2.5-pro")
        self.assertTrue(seen["narrative_schema_has_no_score_fields"])


class DegradedModeTest(unittest.TestCase):
    """A corpus or model outage must not lose the deterministic assessment."""

    def _agent_with(self, retrieval, narrative):
        agent = VendorQualificationAgent()

        def fake_grounded(**kwargs):
            if isinstance(retrieval, Exception):
                raise retrieval
            return retrieval

        async def fake_narrative(**kwargs):
            if isinstance(narrative, Exception):
                raise narrative
            return narrative

        agent.generate_grounded = fake_grounded
        agent.generate_with_rag = fake_narrative
        return agent

    def test_a_retrieval_outage_falls_back_to_the_local_policy_store(self):
        """The corpus is unreachable: the run still cites the sections the
        decision was made under, and the dossier discloses which store served
        the pass rather than claiming a corpus it never reached."""
        agent = self._agent_with(RuntimeError("corpus unreachable"), NARRATIVE_RESULT)
        dossier = _dossier_from(_events(agent, complete_form()))

        self.assertTrue(dossier["rag_retrieval_citations"])
        self.assertTrue(dossier["rag_available"])
        meta = dossier["retrieval_metadata"]
        self.assertEqual(meta["source"], "local-policy-corpus")
        self.assertTrue(meta["retrieved_at"])
        self.assertGreaterEqual(meta["passages"], 1)
        # The score is unaffected and still correct, and the content guardrails
        # still hold. A corpus outage is reported through retrieval_metadata,
        # not by pretending the assessment is unsound.
        self.assertTrue(dossier["mandatory_edd_triggered"])
        self.assertEqual(dossier["assigned_risk_tier"], "High Risk (EDD)")
        self.assertGreaterEqual(dossier["weighted_risk_score"], 2.50)
        self.assertTrue(dossier["guardrail_check_passed"])
        self.assertEqual(contract.check_invariants(dossier), [])

    def test_local_store_citations_are_backed_by_readable_text(self):
        """An outage used to leave zero citations; now the local store answers.
        But never as a silent stand-in: every citation still carries the quote
        and section a reviewer can read, and the store that served them is
        named in retrieval_metadata."""
        agent = self._agent_with(RuntimeError("corpus unreachable"), NARRATIVE_RESULT)
        dossier = _dossier_from(_events(agent, complete_form()))

        summary = dossier["citation_summary"]
        self.assertGreater(summary["total"], 0)
        self.assertGreater(summary["verified"], 0)
        self.assertEqual(
            dossier["retrieval_metadata"]["source"], "local-policy-corpus")
        for citation in dossier["rag_retrieval_citations"]:
            if citation["verified"]:
                self.assertTrue(citation["quote"])
                self.assertTrue(citation["source_uri"])
                self.assertTrue(citation["section"])

    def test_a_narrative_outage_still_produces_a_scored_dossier(self):
        agent = self._agent_with(RETRIEVAL_RESULT, LLMParseError("bad json"))
        events = _events(agent, complete_form())
        dossier = _dossier_from(events)

        self.assertIn("error", [e["type"] for e in events])
        self.assertTrue(dossier["mandatory_edd_triggered"])
        self.assertEqual(dossier["assigned_risk_tier"], "High Risk (EDD)")
        # The score sheet is tier-derived, so the outage cannot blank it: EDD
        # reads 55.0, assessed, flagged for MLRO clearance on the sheet itself.
        self.assertEqual(dossier["appendix_f_score"]["total_score"], 55.0)
        self.assertTrue(dossier["appendix_f_score"]["assessed"])
        self.assertEqual(dossier["appendix_f_score"]["status"], "FAILED")
        self.assertEqual(dossier["assessment_narrative"], "")
        # NEEDS_EDD, not rejection and not an "unassessed" scolding: this
        # vendor carries a mandatory trigger, and the sheet's flag is stated
        # where the officer reads the sheet.
        self.assertEqual(dossier["qualification_status"], "NEEDS_EDD")
        self.assertTrue(dossier["guardrail_check_passed"])
        self.assertNotIn("has not been assessed", dossier["next_action"])

    def test_the_appendix_sheet_comes_from_the_risk_tier_whatever_the_narrative_says(self):
        """The model's category opinions are a cross-check, not the payload's
        number: a strong narrative cannot lift a sheet the tier flagged."""
        agent = self._agent_with(RETRIEVAL_RESULT, NARRATIVE_RESULT)
        dossier = _dossier_from(_events(agent, complete_form()))

        appendix = dossier["appendix_f_score"]
        self.assertTrue(appendix["assessed"])
        self.assertEqual(appendix["total_score"], 55.0)
        self.assertEqual(appendix["status"], "FAILED")
        self.assertNotIn("has not been assessed", dossier["next_action"])

    def test_retrieval_that_never_conforms_falls_back_to_the_local_store(self):
        """A response that never parsed is the same as no response: the local
        store answers instead of the dossier reporting empty citations."""
        agent = self._agent_with({"data": None, "chunks": []}, NARRATIVE_RESULT)
        dossier = _dossier_from(_events(agent, complete_form()))
        self.assertTrue(dossier["rag_retrieval_citations"])
        self.assertTrue(dossier["rag_available"])
        self.assertEqual(
            dossier["retrieval_metadata"]["source"], "local-policy-corpus")


class OnboardingFlagPayloadTest(unittest.TestCase):
    """`onboarding_flags` must reach Tool 2 on every dossier, including the
    unassessed one, and must not perturb the assessment."""

    def _stubbed(self):
        agent = VendorQualificationAgent()
        agent.generate_grounded = lambda **kwargs: {"data": RETRIEVAL_RESULT["data"],
                                                    "chunks": RETRIEVAL_RESULT["chunks"]}
        agent.generate_with_rag = lambda **kwargs: _async_none(NARRATIVE_RESULT)
        return agent

    def test_the_flags_are_present_on_an_assessed_dossier(self):
        dossier = _dossier_from(_events(self._stubbed(), complete_form()))
        self.assertEqual(dossier["onboarding_flags"], {
            "bank_callback_verification_required": True,
            "audited_financials_verified": True,
        })

    def test_the_flags_are_present_on_a_rejected_incomplete_dossier(self):
        # Tool 2 must still learn that the bank was never called back on, even
        # for a submission that never got far enough to hold bank details.
        dossier = _dossier_from(_events(self._stubbed(), {}))
        self.assertEqual(dossier["qualification_status"], "REJECTED_INCOMPLETE")
        self.assertEqual(dossier["onboarding_flags"], {
            "bank_callback_verification_required": True,
            "audited_financials_verified": False,
        })

    def test_an_unverified_audit_does_not_change_the_verdict(self):
        agent = self._stubbed()
        with_report = _dossier_from(_events(agent, complete_form()))

        other = VendorQualificationAgent()
        other.generate_grounded = agent.generate_grounded
        other.generate_with_rag = agent.generate_with_rag
        without_report = _dossier_from(_events(other, dict(
            complete_form(), financial_statements_audited="No")))

        for field in ("qualification_status", "weighted_risk_score",
                      "assigned_risk_tier", "mandatory_edd_triggered",
                      "trigger_reasons", "next_action"):
            self.assertEqual(with_report[field], without_report[field], msg=field)
        self.assertFalse(without_report["onboarding_flags"]["audited_financials_verified"])


class DeterminismTest(unittest.TestCase):
    def test_the_same_submission_scores_identically_twice(self):
        agent = VendorQualificationAgent()
        agent.generate_grounded = lambda **kwargs: {"data": None, "chunks": []}
        agent.generate_with_rag = lambda **kwargs: _async_none()

        first = _dossier_from(_events(agent, complete_form()))
        second = _dossier_from(_events(agent, complete_form()))

        for field in ("weighted_risk_score", "assigned_risk_tier",
                      "mandatory_edd_triggered", "trigger_reasons", "vendor_id"):
            self.assertEqual(first[field], second[field], msg=field)

    def test_the_vendor_id_is_stable_for_the_same_legal_name(self):
        self.assertEqual(
            _vendor_id("Apex Gulf Technical Solutions LLC"),
            _vendor_id("  apex gulf technical solutions llc  "),
        )
        self.assertNotEqual(
            _vendor_id("Apex Gulf Technical Solutions LLC"),
            _vendor_id("Apex Gulf Logistics LLC"),
        )

    def test_the_vendor_id_looks_like_a_control_number(self):
        vendor_id = _vendor_id("Apex Gulf Technical Solutions LLC")
        self.assertTrue(vendor_id.startswith("VND-"))
        self.assertRegex(vendor_id, r"^VND-\d{4}-\d{5}$")


class PolicyAnchorTest(unittest.TestCase):
    """The evidence pool must contain the policy sections, not just whatever one
    broad query happened to return. Across live runs the same submission came
    back grounded 1-of-14 and then 0-of-6 purely on retrieval luck."""

    ANCHOR_CHUNK = {
        "uri": "https://drive.google.com/file/d/1POLICY/view",
        "text": ("Risk Scoring Matrix The Group assigns a weighted risk score "
                 "on a scale of 1.00 to 3.00 based on the Risk Factors."),
        "score": 0.7,
    }

    def test_anchor_chunks_are_merged_into_the_evidence_pool(self):
        agent = VendorQualificationAgent()
        seen = {"n": 0}

        def distinct(**kwargs):
            seen["n"] += 1
            return {"data": {"verbatim_text": ["..."]},
                    "chunks": [{"uri": f"uri-{seen['n']}",
                                "text": f"policy section {seen['n']}",
                                "score": 0.7}]}

        agent.generate_grounded = distinct
        chunks = agent._anchor_policy_text([])
        self.assertEqual(len(chunks), len(vq_schemas.POLICY_ANCHOR_QUERIES))
        self.assertTrue(all(c["text"] for c in chunks))

    def test_anchor_chunks_are_deduplicated(self):
        agent = VendorQualificationAgent()
        agent.generate_grounded = lambda **kwargs: {
            "data": {"verbatim_text": ["..."]}, "chunks": [self.ANCHOR_CHUNK]}

        chunks = agent._anchor_policy_text([dict(self.ANCHOR_CHUNK)])
        self.assertEqual(len(chunks), 1)

    def test_an_empty_anchor_response_adds_nothing(self):
        agent = VendorQualificationAgent()
        agent.generate_grounded = lambda **kwargs: {
            "data": {"verbatim_text": []}, "chunks": []}
        self.assertEqual(agent._anchor_policy_text([]), [])

    def test_the_narrative_is_retried_before_the_citations_are_given_up(self):
        """One transient schema parse failure cost a live run its entire citation
        set, turning a dossier with grounded policy references into one with
        none. The deterministic half is unaffected, so the narrative is worth a
        second attempt."""
        agent = self._narrative_agent()
        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMParseError("truncated json")
            return {"assessment_narrative": "ok"}

        agent.generate_with_rag = flaky
        form = complete_form()
        bag = signal_lib.build_signals(form)
        assessment, override = agent.assess(bag)

        narrative = asyncio.run(
            agent._generate_narrative(form, bag, assessment, override))
        self.assertIsNotNone(narrative)
        self.assertEqual(calls["n"], 2)

    def test_the_narrative_gives_up_after_two_attempts(self):
        agent = self._narrative_agent()
        calls = {"n": 0}

        async def always_fails(**kwargs):
            calls["n"] += 1
            raise LLMParseError("truncated json")

        agent.generate_with_rag = always_fails
        form = complete_form()
        bag = signal_lib.build_signals(form)
        assessment, override = agent.assess(bag)

        self.assertIsNone(asyncio.run(
            agent._generate_narrative(form, bag, assessment, override)))
        self.assertEqual(calls["n"], 2, msg="retried forever, not once")

    def _narrative_agent(self):
        """An agent with Phase A retrieval stubbed and nothing else.

        `generate_grounded` has to be replaced too: the retry tests otherwise
        fall through to the real Vertex client, which is why the suite went from
        0.03s to 230s once the retry was added.
        """
        agent = VendorQualificationAgent()
        agent.generate_grounded = lambda **kwargs: {
            "data": {"verbatim_text": []}, "chunks": [POLICY_CHUNK]}
        return agent

    def test_a_lost_narrative_still_yields_a_scored_dossier(self):
        agent = self._narrative_agent()
        agent.generate_with_rag = _raise_on_narrative
        dossier = _dossier_from(_events(agent, complete_form()))
        self.assertEqual(dossier["qualification_status"], "NEEDS_EDD")
        self.assertEqual(dossier["weighted_risk_score"], 2.5)
        self.assertEqual(dossier["assigned_risk_tier"], "High Risk (EDD)")
        self.assertTrue(dossier["mandatory_edd_triggered"])
        self.assertTrue(dossier["trigger_reasons"])

    def test_the_appendix_f_sheet_is_derived_from_the_risk_tier(self):
        """Three tiers, three reference totals — all assessed, no narrative
        involved: the sheet restates the band table's decision, so it can
        neither come back blank nor disagree with the score it came from."""
        expectations = {
            vq_schemas.TIER_LOW: (88.5, "PASSED"),
            vq_schemas.TIER_MEDIUM: (76.0, "PASSED"),
            vq_schemas.TIER_HIGH: (55.0, "FAILED"),
        }
        for tier, (total, status) in expectations.items():
            sheet = vq_schemas.appendix_f_scores(
                vq_schemas.appendix_f_for_tier(tier))
            self.assertEqual(sheet["total_score"], total, tier)
            self.assertEqual(sheet["status"], status, tier)
            # Every tier raw sits at or above the category floor, so the sheet
            # fails on its total alone rather than inventing category failures.
            self.assertEqual(sheet["failed_categories"], [], tier)

    def test_a_narrative_outage_never_leaves_the_sheet_unassessed(self):
        """What an outage used to do: blank the sheet to 0.0, fail the
        guardrail, and warn the officer off onboarding over a model hiccup.
        The sheet is tier-derived now, so the outage cannot touch it — and a
        flagged sheet on an EDD vendor still reads NEEDS_EDD, with the flag
        stated in the sheet's own rationale."""
        agent = self._narrative_agent()
        agent.generate_with_rag = _raise_on_narrative
        dossier = _dossier_from(_events(agent, complete_form()))

        appendix = dossier["appendix_f_score"]
        self.assertTrue(appendix["assessed"])
        self.assertEqual(appendix["total_score"], 55.0)
        self.assertEqual(appendix["status"], "FAILED")
        self.assertIn("FLAGGED", appendix["rationale"])
        self.assertEqual(dossier["qualification_status"], "NEEDS_EDD")
        self.assertNotEqual(dossier["qualification_status"], "REJECTED_HIGH_RISK")
        self.assertTrue(dossier["guardrail_check_passed"])
        self.assertNotIn("has not been assessed", dossier["next_action"])

    def test_a_corpus_outage_during_anchoring_is_survivable(self):
        agent = VendorQualificationAgent()

        def boom(**kwargs):
            raise RuntimeError("corpus down")

        agent.generate_grounded = boom
        self.assertEqual(agent._anchor_policy_text([]), [])

    def test_anchoring_never_discards_what_the_main_pass_already_found(self):
        agent = VendorQualificationAgent()
        agent.generate_grounded = lambda **kwargs: {
            "data": {"verbatim_text": ["..."]}, "chunks": []}
        original = [{"uri": "u", "text": "main pass text", "score": 0.1}]
        chunks = agent._anchor_policy_text(original)
        self.assertIn(original[0], chunks)


class DisqualificationTest(unittest.TestCase):
    def test_a_prohibited_jurisdiction_is_rejected_not_escalated(self):
        retrieval = {
            "data": {
                "jurisdiction_tier": "prohibited",
                "jurisdiction_basis": "Prohibited Jurisdiction.",
                "business_category": "Undetermined",
                "rules": [],
            },
            "chunks": [POLICY_CHUNK],
        }
        agent = VendorQualificationAgent()
        agent.generate_grounded = lambda **kwargs: retrieval
        agent.generate_with_rag = lambda **kwargs: _async_none()

        dossier = _dossier_from(_events(agent, complete_form()))
        self.assertEqual(dossier["qualification_status"], "REJECTED_HIGH_RISK")
        self.assertIn("Prohibited", dossier["rejection_reason"])
        self.assertIn("Do not route to the Vendor Onboarding Agent", dossier["next_action"])

    def test_a_failed_appendix_f_sheet_without_an_edd_trigger_is_rejected(self):
        """The precedence the sheet's tier needs: with no mandatory EDD
        trigger behind it, a failed sheet blocks the engagement rather than
        escalating it — the tier-derived EDD sheet (55.0) cannot pass, and
        commissioning diligence would not change it. The dossier says why."""
        agent = VendorQualificationAgent()
        sheet = vq_schemas.appendix_f_scores(
            vq_schemas.appendix_f_for_tier(vq_schemas.TIER_HIGH))
        override = trigger_lib.MandatoryEddResult(triggered=False, triggers=[])

        status, reason = agent._status_for(
            override, signal_lib.build_signals(complete_form()), sheet, True)

        self.assertEqual(status, "REJECTED_HIGH_RISK")
        self.assertIn("Appendix F", reason)
        self.assertIn("55.0", reason)


async def _async_none(value=None):
    return value


if __name__ == "__main__":
    unittest.main()
