"""
FastAPI surface: routes, auth, request models, SSE, and the contract audit.

The agent and the deterministic engine are stubbed here; these tests assert on
the wiring around them, not on the risk arithmetic, which tests/ owns.
"""

import asyncio
import json
import unittest
from unittest import mock

import main

from tests.fixtures import complete_form, without

DOSSIER = {
    "vendor_id": "VND-2026-00001",
    "vendor_name": "Apex Gulf Technical Solutions LLC",
    "qualification_status": "NEEDS_EDD",
    "weighted_risk_score": 2.5,
    "assigned_risk_tier": "High Risk (EDD)",
    "mandatory_edd_triggered": True,
    "trigger_reasons": ["Strategic spend"],
    "company_profile": {},
    "ownership_summary": {},
    "jurisdiction_assessment": {},
    "weighted_factors": [],
    "appendix_f_score": {},
    "required_controls": [],
    "rag_retrieval_citations": [],
    "citation_summary": {"total": 0, "verified": 0, "unverified": 0,
                         "sources": [], "sections": []},
    "rag_available": True,
    "rejection_reason": "",
    "next_action": "Proceed.",
    "assessment_narrative": "n",
    "assessment_notes": [],
    "guardrail_check_passed": True,
    "timestamp": "2026-09-30T00:00:00Z",
}


class RouteTest(unittest.TestCase):
    def test_the_documented_routes_are_registered(self):
        paths = {route.path for route in main.app.routes}
        for expected in (
            "/health",
            "/api/vendors/qualifications/matrix",
            "/api/vendors/qualifications/validate",
            "/api/vendors/qualifications/score",
            "/api/vendors/qualifications/guidance",
            "/api/vendors/qualifications/prequalification",
            "/api/vendors/qualifications/qualify",
            "/api/vendors/qualifications/stream",
        ):
            self.assertIn(expected, paths)

    def test_every_vendor_route_that_takes_a_submission_is_authenticated(self):
        """No route that reads a vendor submission may be reachable without a
        token. /health and /matrix are the only public ones, and neither
        receives a submission."""
        public = {"/health", "/api/vendors/qualifications/matrix"}
        for route in main.app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/api/vendors/qualifications"):
                continue
            if path in public:
                continue
            with self.subTest(path=path):
                names = {dep.call for dep in route.dependant.dependencies}
                self.assertIn(main.verify_firebase_token, names)

    def test_cors_allows_the_frontend_origins(self):
        middleware = [m for m in main.app.user_middleware
                      if m.cls.__name__ == "CORSMiddleware"]
        self.assertTrue(middleware)
        allowed = {o for o in (main.ALLOWED_ORIGINS or [])}
        self.assertTrue(any("localhost" in o for o in allowed), msg=str(allowed))


class HealthTest(unittest.TestCase):
    def test_health_reports_configuration_rather_than_raising(self):
        result = asyncio.run(main.health_check())
        self.assertEqual(result["status"], "healthy")
        self.assertIn("rag_corpus_configured", result["config"])
        self.assertIn("auth_enabled", result["config"])

    def test_health_does_not_touch_vertex_ai(self):
        with mock.patch.object(main, "get_agent", side_effect=AssertionError("no RAG")):
            result = asyncio.run(main.health_check())
        self.assertEqual(result["status"], "healthy")


class MatrixTest(unittest.TestCase):
    def test_the_matrix_publishes_the_deterministic_matrix(self):
        data = asyncio.run(main.matrix_endpoint())
        self.assertEqual(sum(f["weight"] for f in data["factors"]), 1.0)
        self.assertEqual(len(data["tier_bands"]), 3)
        self.assertTrue(data["triggers"])
        self.assertIn("appendix_f", data)

    def test_the_matrix_tier_bands_are_ordered_and_complete(self):
        data = asyncio.run(main.matrix_endpoint())
        bounds = [b["upper_bound"] for b in data["tier_bands"]]
        self.assertEqual(bounds, sorted(bounds))
        self.assertEqual(bounds[-1], 3.0)
        for band in data["tier_bands"]:
            self.assertTrue(band["due_diligence_level"])
            self.assertTrue(band["refresh_cycle"])


class ScoreEndpointTest(unittest.TestCase):
    """POST /score is the wizard's live gauge. It must be pure, fast, and it
    must not contradict the engine."""

    def test_a_benign_vendor_scores_low_with_no_trigger(self):
        data = _score(_form_with(**{"jurisdiction_risk_tier": "low",
                                    "ownership_structure": "simple",
                                    "estimated_spend_aed": 250_000}))
        self.assertTrue(data["is_complete"])
        self.assertFalse(data["overridden"])
        self.assertEqual(data["assigned_risk_tier"], data["base_tier"])
        self.assertEqual(len(data["factors"]), 8)

    def test_a_strategic_spend_override_moves_the_dd_level_with_the_tier(self):
        """The gauge must not read "High Risk (EDD)" next to "Standard DD, every
        2 years". All three come from the final score."""
        data = _score(_form_with(estimated_spend_aed=2_500_000))
        self.assertTrue(data["overridden"])
        self.assertEqual(data["assigned_risk_tier"], "High Risk (EDD)")
        self.assertLess(data["base_score"], data["weighted_risk_score"])
        self.assertIn("Enhanced", data["due_diligence_level"])
        self.assertEqual(data["refresh_cycle"], "Annual Refresh")

    def test_the_preview_makes_no_model_call(self):
        with mock.patch.object(main, "get_agent",
                               side_effect=AssertionError("no agent")):
            data = _score(complete_form())
        self.assertIn("weighted_risk_score", data)

    def test_an_incomplete_submission_is_not_scored(self):
        """Guardrail: no score for an incomplete form. With every factor on its
        neutral anchor the weighted average is 1.90, which rendered as a Medium
        Risk verdict for a form nobody had filled in."""
        data = _score(without(complete_form(), "trade_license_no", "ubos"))
        self.assertFalse(data["is_complete"])
        self.assertEqual(data["qualification_status"], "REJECTED_INCOMPLETE")
        self.assertEqual(data["guardrail_message"],
                         "Fill mandatory fields to compute risk score.")
        self.assertIsNone(data["weighted_risk_score"])
        self.assertIsNone(data["base_score"])
        self.assertIsNone(data["assigned_risk_tier"])
        self.assertIsNone(data["base_tier"])
        self.assertEqual(data["factors"], [])
        self.assertEqual(data["triggers"], [])
        self.assertGreater(data["missing_count"], 0)

    def test_an_empty_form_does_not_resolve_to_the_neutral_anchor_average(self):
        """The specific regression: a blank form used to report 1.90 /
        Medium Risk (CDD), which is the arithmetic of the neutral anchors."""
        data = _score({})
        self.assertEqual(data["qualification_status"], "REJECTED_INCOMPLETE")
        self.assertIsNone(data["weighted_risk_score"])
        self.assertIsNone(data["assigned_risk_tier"])
        self.assertEqual(data["due_diligence_level"], "")

    def test_a_complete_submission_reports_an_assessed_status(self):
        data = _score(complete_form())
        self.assertEqual(data["qualification_status"], "ASSESSED")
        self.assertEqual(data["guardrail_message"], "")
        self.assertIsNotNone(data["weighted_risk_score"])


class GuidanceEndpointTest(unittest.TestCase):
    """POST /guidance backs the wizard's Live Policy Guidance panel. It retrieves
    policy text and must never score or assess."""

    def _guidance(self, form):
        return asyncio.run(
            main.guidance_endpoint({"form": form}, _auth())
        )["data"]

    def test_an_incomplete_form_still_gets_policy_text(self):
        """The panel's whole purpose is guidance while the officer is still
        typing, so the completeness gate must not block it."""
        with mock.patch.object(main, "get_agent", _stub_agent(
            retrieval={"jurisdiction_tier": "high", "category": "Contractor",
                       "rules": [{"rule_applied": "x"}]},
            citations=[_citation(verified=True)],
        )):
            data = self._guidance({"legal_name": "Only One Field"})
        self.assertTrue(data["available"])
        self.assertEqual(data["cited_count"], 1)
        self.assertEqual(data["total_count"], 1)

    def test_it_returns_no_score_or_tier(self):
        with mock.patch.object(main, "get_agent", _stub_agent(
            retrieval={"jurisdiction_tier": "low", "rules": []},
        )):
            data = self._guidance(complete_form())
        for forbidden in ("weighted_risk_score", "assigned_risk_tier",
                          "base_score", "status"):
            self.assertNotIn(forbidden, data)

    def test_a_corpus_outage_is_reported_not_raised(self):
        """Guidance is advisory. An unreachable corpus must not break the form."""
        class _Broken:
            def retrieve_policy_facts(self, form):
                raise RuntimeError("corpus unavailable")

        with mock.patch.object(main, "get_agent", lambda: _Broken()):
            data = self._guidance(complete_form())
        self.assertFalse(data["available"])
        self.assertIn("corpus unavailable", data["reason"])
        self.assertEqual(data["citations"], [])

    def test_it_does_not_call_the_narrative_model(self):
        """Guidance is retrieval only. Running the pro model per keystroke would
        be both slow and pointless."""

        class _NoPro:
            def __init__(self):
                self.narrative_calls = 0

            def retrieve_policy_facts(self, form):
                return {"retrieval": {"rules": []}, "citations": []}

        agent = _NoPro()
        with mock.patch.object(main, "get_agent", lambda: agent):
            self._guidance(complete_form())
        self.assertEqual(agent.narrative_calls, 0)


class PrequalificationEndpointTest(unittest.TestCase):
    """POST /prequalification is the Appendix F 0-100 sheet, distinct from the
    1.00-3.00 risk score."""

    def test_an_incomplete_form_is_not_prequalified(self):
        data = asyncio.run(
            main.prequalification_endpoint({"form": {}}, _auth())
        )["data"]
        self.assertFalse(data["available"])
        self.assertFalse(data["assessed"])
        # One entry per blocking field on an empty submission, including the
        # Item 20 certifications and Item 40a audit declaration. Exact, so a new
        # blocking field has to be acknowledged here rather than drift in.
        self.assertEqual(data["missing_count"], 38)

    def test_a_complete_form_returns_the_weighted_sheet(self):
        narrative = {"appendix_f_assessment": {
            "financial_standing": 90.0,
            "technical_capability": 95.0,
            "quality_hse": 85.0,
            "rationale": "Strong audited accounts.",
        }}
        with mock.patch.object(main, "get_agent", _stub_agent(narrative=narrative)):
            data = asyncio.run(
                main.prequalification_endpoint(
                    {"form": complete_form()}, _auth())
            )["data"]
        self.assertTrue(data["available"])
        self.assertTrue(data["assessed"])
        self.assertEqual(data["status"], "PASSED")
        self.assertEqual(data["total_score"], 90.3)
        self.assertAlmostEqual(data["financial_standing_score"], 31.5, places=1)
        self.assertAlmostEqual(data["technical_capability_score"], 33.3, places=1)
        self.assertAlmostEqual(data["quality_hse_score"], 25.5, places=1)
        self.assertEqual(data["weights"],
                         {"financial_standing": 35.0,
                          "technical_capability": 35.0,
                          "quality_hse": 30.0})

    def test_an_absent_narrative_is_not_a_failed_sheet(self):
        """A model outage must not report the vendor as failing Appendix F."""
        with mock.patch.object(main, "get_agent", _stub_agent(narrative=None)):
            data = asyncio.run(
                main.prequalification_endpoint(
                    {"form": complete_form()}, _auth())
            )["data"]
        self.assertTrue(data["available"])
        self.assertFalse(data["assessed"])


class ValidateEndpointTest(unittest.TestCase):
    def test_a_complete_submission_is_reported_complete(self):
        response = _validate(complete_form())
        self.assertTrue(response["data"]["report"]["is_complete"])
        self.assertEqual(response["data"]["report"]["missing_fields"], [])
        self.assertEqual(response["data"]["report"]["missing_count"], 0)
        self.assertEqual(response["status"], "success")

    def test_an_empty_submission_lists_the_gaps(self):
        response = _validate({})
        report = response["data"]["report"]
        self.assertFalse(report["is_complete"])
        self.assertGreater(report["missing_count"], 20)
        self.assertTrue(report["missing_fields"])
        self.assertEqual(response["status"], "success")

    def test_the_missing_data_request_names_the_form(self):
        response = _validate({})
        request = response["data"]["missing_data_request"]
        self.assertIn("NH-PQF-001", request["message"])
        self.assertTrue(request["fields"])
        self.assertEqual(response["data"]["policy"]["form"],
                         main.FORM_DOCUMENT)

    def test_validation_makes_no_model_call(self):
        with mock.patch.object(main, "get_agent",
                               side_effect=AssertionError("no agent")):
            response = _validate(complete_form())
        self.assertTrue(response["data"]["report"]["is_complete"])


class QualifyEndpointTest(unittest.TestCase):
    def test_a_dossier_is_returned_under_the_status_envelope(self):
        with mock.patch.object(main, "get_agent", new=_agent_returning(DOSSIER)):
            response = asyncio.run(
                main.qualify_endpoint(complete_form(), _auth()))

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["data"]["assigned_risk_tier"], "High Risk (EDD)")
        self.assertIn("contract_audit", response["data"])

    def test_a_contract_violation_is_reported_and_downgrades_the_guardrail(self):
        broken = dict(DOSSIER)
        broken["assigned_risk_tier"] = "Low Risk (SDD)"
        with mock.patch.object(main, "get_agent", new=_agent_returning(broken)):
            response = asyncio.run(
                main.qualify_endpoint(complete_form(), _auth()))

        audit = response["data"]["contract_audit"]
        self.assertFalse(audit["valid"])
        self.assertEqual(response["contract"]["valid"], False)
        self.assertTrue(audit["violations"])
        # A violation must not leave a true claim standing.
        self.assertFalse(response["data"]["guardrail_check_passed"])


class StreamEndpointTest(unittest.TestCase):
    def test_events_are_framed_as_sse_data_lines(self):
        with mock.patch.object(main, "get_agent", new=_agent_returning(DOSSIER)):
            response = asyncio.run(
                main.stream_endpoint(complete_form(), _auth()))
            body = _collect(response)

        self.assertTrue(body)
        for line in body.splitlines():
            if not line:
                continue  # SSE frame separator
            self.assertTrue(line.startswith("data: "), msg=line)
            json.loads(line[len("data: "):])

    def test_the_stream_ends_with_a_dossier_and_the_contract_audit(self):
        with mock.patch.object(main, "get_agent", new=_agent_returning(DOSSIER)):
            response = asyncio.run(
                main.stream_endpoint(complete_form(), _auth()))
            events = _events(_collect(response))

        self.assertEqual(events[-1]["type"], "dossier")
        self.assertIn("contract_audit", events[-1]["dossier"])
        self.assertIn("violations", events[-1]["dossier"]["contract_audit"])

    def test_a_failure_mid_stream_is_reported_as_an_error_event(self):
        async def exploding(form):
            raise RuntimeError("corpus down")
            yield  # pragma: no cover - makes this an async generator

        with mock.patch.object(main, "get_agent", new=_agent_returning(stream=exploding)):
            response = asyncio.run(
                main.stream_endpoint(complete_form(), _auth()))
            events = _events(_collect(response))

        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("corpus down", events[-1]["message"])

    def test_the_stream_response_is_not_buffered(self):
        with mock.patch.object(main, "get_agent", new=_agent_returning(DOSSIER)):
            response = asyncio.run(
                main.stream_endpoint(complete_form(), _auth()))
        self.assertEqual(response.media_type, "text/event-stream")
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertEqual(response.headers["x-accel-buffering"], "no")


class AuthTest(unittest.TestCase):
    def test_auth_fails_closed_when_firebase_is_unavailable(self):
        with mock.patch.object(main, "AUTH_DISABLED", False), \
             mock.patch.object(main, "_FIREBASE_READY", False):
            with self.assertRaises(Exception) as caught:
                asyncio.run(main.verify_firebase_token(None))
        self.assertEqual(caught.exception.status_code, 503)

    def test_a_missing_bearer_token_is_401(self):
        with mock.patch.object(main, "AUTH_DISABLED", False), \
             mock.patch.object(main, "_FIREBASE_READY", True):
            for header in (None, "", "Token abc", "abc"):
                with self.subTest(header=header):
                    with self.assertRaises(Exception) as caught:
                        asyncio.run(main.verify_firebase_token(header))
                    self.assertEqual(caught.exception.status_code, 401)

    def test_an_invalid_token_is_401(self):
        with mock.patch.object(main, "AUTH_DISABLED", False), \
             mock.patch.object(main, "_FIREBASE_READY", True), \
             mock.patch.object(main.auth, "verify_id_token",
                               side_effect=ValueError("expired")):
            with self.assertRaises(Exception) as caught:
                asyncio.run(main.verify_firebase_token("Bearer token"))
        self.assertEqual(caught.exception.status_code, 401)

    def test_auth_disabled_is_only_honoured_when_explicitly_set(self):
        with mock.patch.object(main, "AUTH_DISABLED", True):
            result = asyncio.run(main.verify_firebase_token(None))
        self.assertEqual(result["uid"], "auth-disabled")

    def test_a_valid_token_passes_through(self):
        with mock.patch.object(main, "AUTH_DISABLED", False), \
             mock.patch.object(main, "_FIREBASE_READY", True), \
             mock.patch.object(main.auth, "verify_id_token",
                               return_value={"uid": "u1"}):
            result = asyncio.run(main.verify_firebase_token("Bearer good"))
        self.assertEqual(result["uid"], "u1")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _auth():
    return {"uid": "test", "email": "test@example.com"}


def _form_with(**overrides):
    form = complete_form()
    form.update(overrides)
    return form


def _score(form):
    return asyncio.run(main.score_endpoint({"form": form}, _auth()))["data"]


def _validate(form):
    return asyncio.run(main.validate_endpoint({"form": form}, _auth()))


def _collect(response):
    async def drain():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)
    return asyncio.run(drain())


def _events(body):
    """Parse an SSE body into payloads. Blank lines are frame separators, not
    data, so they are skipped rather than treated as a malformed frame."""
    return [json.loads(line[len("data: "):])
            for line in body.splitlines() if line.startswith("data: ")]


def _citation(verified=False, quote="Policy text.", rule="A rule."):
    """A citation_lib.Citation stand-in with a to_dict()."""
    class _C:
        def __init__(self):
            self.document = "Procurement Policy_Updated_2.pdf"
            self.section = "Risk Factors"
            self.rule_applied = rule
            self.quote = quote
            self.source_uri = "gs://corpus/doc"
            self.score = 0.7
            self.verified = verified
            self.matched_trigger = None
            self.factor = None

        def to_dict(self):
            return {
                "document": self.document,
                "section": self.section,
                "rule_applied": self.rule_applied,
                "quote": self.quote,
                "source_uri": self.source_uri,
                "score": self.score,
                "verified": self.verified,
                "matched_trigger": self.matched_trigger,
                "factor": self.factor,
            }

    return _C()


def _stub_agent(retrieval=None, citations=None, narrative=None):
    """A stand-in for get_agent() covering the retrieval-only and narrative
    endpoints. `retrieve_policy_facts` is synchronous and blocks on Vertex AI,
    exactly as the real one does."""

    class _Stub:
        model_name = "gemini-2.5-pro"

        def retrieve_policy_facts(self, form):
            return {
                "retrieval": retrieval or {},
                "citations": list(citations or []),
            }

        async def _generate_narrative(self, form, signal_bag, assessment,
                                      override):
            return narrative

    return lambda: _Stub()


def _agent_returning(dossier=None, events=None, stream=None):
    """A stand-in for get_agent().

    Pass a `dossier` to have both qualify() and qualify_stream() report it, an
    explicit `events` list to drive the stream, or a `stream` async generator
    factory to simulate a mid-stream failure.
    """
    if events is None and dossier is not None:
        events = [{"type": "stage", "stage": "stub"},
                  {"type": "dossier", "dossier": dossier}]

    class _Stub:
        async def qualify(self, form):
            return dossier

        async def qualify_stream(self, form):
            if stream is not None:
                async for event in stream(form):
                    yield event
                return
            for event in events:
                yield event

    return lambda: _Stub()


if __name__ == "__main__":
    unittest.main()
