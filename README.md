# Vendor Qualification & Risk Tiering Agent — Tool 1 backend

FastAPI service that turns an NH-PQF-001 vendor prequalification submission into
the `output_schema.md` dossier: a deterministic weighted risk score, a risk tier,
mandatory Enhanced Due Diligence triggers, a scored Appendix F sheet, and
policy citations.

## The one design rule

**The model does not decide anything that matters.**

Two Gemini passes sit either side of a deterministic Python core:

| Phase | Model | Job |
| --- | --- | --- |
| Validation | none | Missing mandatory fields. Short-circuits before any model call. |
| A — retrieval | `gemini-2.5-flash` | Find the policy rules that apply, quoted from the corpus. |
| — scoring | none | Weighted score, tier, EDD triggers, Appendix F arithmetic. |
| B — narrative | `gemini-2.5-pro` | Profile, ownership narrative, Appendix F raw marks, controls, questions. |

`risk_engine.py`, `triggers.py`, `validation.py` and the Appendix F arithmetic in
`agents/vendor_qualification/schemas.py` own every number and every category in
the output. The model is structurally unable to change them: the Phase B response
schema has no `weighted_risk_score` and no `assigned_risk_tier` property to fill
in. Whatever the narrative says is merged underneath the engine's verdict.

## Consequence: a vendor is never scored on a hallucination

An incomplete submission never reaches a model at all. If a mandatory NH-PQF-001
field is missing, the response is `REJECTED_INCOMPLETE` with
`weighted_risk_score: null` and `assigned_risk_tier: null`, plus the
missing-data request. Reporting `1.00 / Low Risk (SDD)` there would be a
fabricated assessment. `tests/test_agent_flow.py::ZeroHallucinationGateTest`
enforces this by making the model methods raise if they are called.

## Scoring

Weighted anchors, each 1.00 (lowest risk) to 3.00 (highest). Weights sum to
`1.00` and are asserted in code.

| Factor | Weight |
| --- | --- |
| Jurisdiction | 0.25 |
| Ownership | 0.20 |
| Government exposure | 0.15 |
| Payment structure | 0.15 |
| Annual spend | 0.10 |
| Nature of business | 0.05 |
| Adverse media | 0.05 |
| Prior relationship | 0.05 |

Tiers: `<= 1.60` Low Risk (SDD), `<= 2.20` Medium Risk (CDD), `<= 3.00`
High Risk (EDD). A mandatory EDD trigger is a **floor** of `2.50`, not a fixed
value, so an already-higher score is left alone.

Statuses:

- `REJECTED_INCOMPLETE` — a mandatory field is missing. No model call.
- `REJECTED_HIGH_RISK` — blocked rather than escalated: Prohibited jurisdiction,
  unresolved sanctions match, adverse-media association, complex ownership, or a
  failed Appendix F sheet.
- `NEEDS_EDD` — at least one mandatory Always-EDD trigger fired.
- `QUALIFIED` — no trigger, no disqualifier, Appendix F passed.

A failed Appendix F outranks an EDD trigger deliberately: a vendor that cannot
pass the prequalification score sheet does not need more risk diligence, it needs
different financials. `output_schema.md` has no "failed the commercial score
sheet" value, so this reports as `REJECTED_HIGH_RISK` with the reason stated in
`rejection_reason`. The enum is not widened, so the Tool 2 contract holds.

## Contract auditing

`contract.py` re-derives the score, tier, EDD floor, trigger reasons, Appendix F
arithmetic and citation evidence from the finished dossier and reports anything
that disagrees. A violation downgrades `guardrail_check_passed` to `False` and is
returned in `contract_audit.violations`. The server never returns a dossier
claiming its guardrails passed when it can prove otherwise.

`jsonschema` is optional at runtime. When installed the dossier is validated
against `schemas/vendor_qualification.schema.json`; when absent, `schema_checked`
is `false` and the deterministic invariants still run.

## Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/health` | no | Config and dependency state. Never touches Vertex AI. |
| GET | `/api/vendors/qualifications/matrix` | no | Factor weights, tier bands, trigger catalogue, Appendix F gate. |
| POST | `/api/vendors/qualifications/validate` | yes | Completeness gate. No model. |
| POST | `/api/vendors/qualifications/score` | yes | Live weighted score for the wizard gauge. No model, no RAG. |
| POST | `/api/vendors/qualifications/qualify` | yes | Full dossier. |
| POST | `/api/vendors/qualifications/stream` | yes | The same, streamed as SSE. |

All submission routes accept either `{"form": {...}}` or a bare form object.

SSE event types, in order: `stage`, `validation`, `jurisdiction`, `category`,
`retrieval`, `score`, `triggers`, `dossier`, `error`. The final `dossier` event
always carries `contract_audit`.

## Citations

A citation is only `verified` when it can be matched to retrieved corpus text
(section heading, or at least 45% token overlap) and carries a quote and a source
URI. A model assertion with no matching text is kept in the dossier but marked
`verified: false` with an empty quote, so the UI can show it as unbacked rather
than drop the reviewer into guessing. A corpus outage yields zero citations and
`rag_available: false`; it does not weaken the score.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit
uvicorn main:app --reload --port 8090
```

`AUTH_DISABLED=true` is for local development only. It turns off Firebase token
verification on every submission route. Without it, auth fails closed: if
Firebase cannot initialise, protected routes return 503 rather than serving an
unauthenticated compliance dossier.

## Tests

```bash
python -m unittest tests.test_risk_engine tests.test_triggers \
  tests.test_validation tests.test_appendix_f tests.test_contract \
  tests.test_agent_flow tests.test_api -v
```

No network and no model. `test_agent_flow.py` and `test_api.py` replace the
generation methods and the agent on the instance, so the whole suite runs
offline.

## Layout

```
main.py                         FastAPI routes, auth, SSE, contract audit
risk_engine.py                  weights, anchors, bands, factor catalogue
triggers.py                     mandatory EDD detection and the score floor
validation.py                   NH-PQF-001 field rules and conditional sections
signals.py                      form -> risk signals, with derivation notes
citations.py                    grounding-backed citation verification
contract.py                     schema + deterministic invariant auditing
agents/base_rag_agent.py        Vertex RAG and structured-output client
agents/vendor_qualification/
  agent.py                      two-phase orchestration, dossier assembly
  schemas.py                    model schemas, Appendix F arithmetic, statuses
  system-prompt.md              grounding rules and role boundaries
schemas/vendor_qualification.schema.json
rag/sensei_rag_config.json      shared corpus, no re-ingestion
tests/
```

## Corpus

Shared with Supplier Contract Review. Nothing is re-ingested:
`projects/test-rag-corpus-project/locations/us-west1/ragCorpora/137359788634800128`
(`rag-data-full-corpus`). Retrieval from `rag.list_files` currently returns HTTP
429 on the `VertexRagDataService requests` quota, which does not block model
retrieval or generation.
