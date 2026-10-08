"""
Vendor Qualification & Risk Tiering Agent — Tool 1 backend.

FastAPI service for the NH-PQF-001 prequalification workflow. Stateless: every
request is a fresh deterministic assessment plus, on the full runs, two RAG
passes against the shared Sensei corpus.

    GET  /health                              service + config
    GET  /api/vendors/qualifications/matrix   weights, bands, trigger catalogue
    POST /api/vendors/qualifications/validate completeness gate only, no LLM
    POST /api/vendors/qualifications/score    deterministic score only, no LLM
    POST /api/vendors/qualifications/qualify  full dossier (output_schema.md)
    POST /api/vendors/qualifications/stream   full dossier, progressive (SSE)

`/validate` and `/score` never call a model. The wizard uses `/score` to drive
the live 1.00-3.00 gauge, and it must keep working when the corpus or the
model is unavailable — the score does not depend on either.

Every dossier that leaves this service is audited by contract.audit(), which
re-derives the arithmetic and downgrades guardrail_check_passed if anything
disagrees. A compliance claim is never asserted on top of numbers that do not
add up.
"""

import base64
import json
import logging
import os
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
import firebase_admin
from firebase_admin import auth, credentials

import contract
import risk_engine
import signals as signal_lib
import triggers as trigger_lib
import validation as validation_lib

import company_profile as profile_lib
import extraction as extraction_lib
import supplier_portal
from services import email as email_service
from services.email import SendResult

# Log the not-yet-logged supplier invite path from the mint route: send_invitation
# already logs to the outbox now, so the mint route does not duplicate it.


# --------------------------------------------------------------------------
# Notification log — presenter outbox.
#
# Tool 1 sends real emails through services/email.py and records a one-off
# notification result on the session/decision, but it exposes no inbox. The
# console therefore cannot show an outbox. This is a small append-only log that
# every outbound Tool 1 notification writes to, so the presenter console has a
# durable outbox to render.
#
# It lives on the backend, not on sessions, because notifications are emitted by
# several code paths (mint, submit, decision, review invite) and re-deriving them
# from sessions would silently drop anything whose send result is not kept on the
# session in the exact shape the console expects.
# --------------------------------------------------------------------------

_NOTIFICATION_LOG_PATH = os.environ.get("NOTIFICATION_LOG_PATH", "").strip() or "./notification_log.db"
import sqlite3
import threading

_log_conn = sqlite3.connect(_NOTIFICATION_LOG_PATH, check_same_thread=False)
_log_conn.execute(
    "CREATE TABLE IF NOT EXISTS notification_log ("
    "seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, subject TEXT NOT NULL,"
    "recipient TEXT, sent BOOLEAN NOT NULL, detail TEXT, token TEXT, link TEXT,"
    "created_at REAL NOT NULL)"
)
_log_conn.commit()
_log_lock = threading.Lock()


def log_notification(
    kind: str,
    subject: str,
    recipient: str,
    sent: bool,
    detail: str,
    token: str = "",
    link: str = "",
) -> None:
    with _log_lock:
        _log_conn.execute(
            "INSERT INTO notification_log (kind, subject, recipient, sent, detail, token, link, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                kind,
                subject,
                recipient or "",
                int(bool(sent)),
                detail or "",
                token or "",
                link or "",
                __import__("time").time(),
            ),
        )
        _log_conn.commit()


async def log_notification_async(
    kind: str,
    subject: str,
    recipient: str,
    sent: bool,
    detail: str,
    token: str = "",
    link: str = "",
) -> None:
    def _run() -> None:
        log_notification(kind, subject, recipient, sent, detail, token, link)

    await asyncio.to_thread(_run)


def list_notification_log() -> list[dict[str, Any]]:
    with _log_lock:
        rows = _log_conn.execute(
            "SELECT seq, kind, subject, recipient, sent, detail, token, link, created_at"
            " FROM notification_log ORDER BY seq DESC LIMIT 200"
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "seq": r[0],
            "kind": r[1],
            "subject": r[2],
            "recipient": r[3],
            "sent": bool(r[4]),
            "detail": r[5],
            "token": r[6],
            "link": r[7],
            "created_at": r[8],
        })
    return out


def notification_exists(kind: str, token: str) -> bool:
    """Has this outbox row already been written? (Chaser idempotency.)"""
    if not token:
        return False
    with _log_lock:
        row = _log_conn.execute(
            "SELECT 1 FROM notification_log WHERE kind = ? AND token = ? LIMIT 1",
            (kind, token),
        ).fetchone()
    return row is not None


def delete_notifications(tokens: list[str]) -> None:
    """Drop the outbox rows belonging to these tokens (demo reset)."""
    if not tokens:
        return
    with _log_lock:
        _log_conn.executemany(
            "DELETE FROM notification_log WHERE token = ?",
            [(t,) for t in tokens],
        )
        _log_conn.commit()


def delete_chaser_notifications() -> None:
    """Drop every autonomous-chaser row (demo reset)."""
    with _log_lock:
        _log_conn.execute("DELETE FROM notification_log WHERE kind LIKE 'chaser-%'")
        _log_conn.commit()


async def _send_email(to: str, subject: str, text: str, html: str | None = None) -> SendResult:
    if html is None:
        html = text.replace("\n", "<br>")
    import asyncio

    def _run() -> SendResult:
        return email_service.send(to, subject, text, html, use_override=False)

    return await asyncio.to_thread(_run)


from agents.vendor_qualification import schemas as vq_schemas
from agents.vendor_qualification.agent import VendorQualificationAgent

PROJECT_ID = os.environ.get("PROJECT_ID", "test-rag-corpus-project")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "sensei-ask-project")
LOCATION = os.environ.get("LOCATION", "us-west1")
RAG_CORPUS = os.environ.get(
    "RAG_CORPUS",
    "projects/test-rag-corpus-project/locations/us-west1/ragCorpora/137359788634800128",
)
POLICY_DOCUMENT = "Procurement Policy_Updated_2.pdf"
FORM_DOCUMENT = "NH-PQF-001-Rev.09 - Vendor Prequalification & FTA Verification Form"

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        # Default covers local dev on the ports this tool and its frontend use.
        "http://localhost:3000,http://localhost:8000,http://localhost:8080,"
        "http://localhost:8089,http://localhost:8090,"
        "http://local.sensei.com:8080,https://sensei-dev.scmdojo.com,"
        "https://sensei.scmdojo.com,https://www.scmdojo.com,"
        "https://scmsensei.ai,https://www.scmsensei.ai,"
        "https://dev.scmsensei.ai",
    ).split(",")
    if origin.strip()
]

logger = logging.getLogger("vendor-qualification")

app = FastAPI(
    title="Vendor Qualification & Risk Tiering Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://[a-zA-Z0-9-]+\.scmsensei\.ai",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# AUTH
# ==========================================================================
# Firebase Admin is initialised lazily and tolerantly: the service must boot
# for local development and for /health without application-default
# credentials. Auth then FAILS CLOSED — protected endpoints return 503 rather
# than serving an unauthenticated compliance dossier, unless the operator
# explicitly sets AUTH_DISABLED.

_FIREBASE_READY = False
_FIREBASE_ERROR = ""

if not firebase_admin._apps:
    try:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {"projectId": FIREBASE_PROJECT_ID})
        _FIREBASE_READY = True
    except Exception as exc:  # noqa: BLE001
        _FIREBASE_ERROR = str(exc)

AUTH_DISABLED = os.environ.get("AUTH_DISABLED", "").strip().lower() in (
    "1", "true", "yes",
)


async def verify_firebase_token(authorization: str = Header(None)) -> Dict[str, Any]:
    """Verify the caller's Firebase ID token.

    Fails closed: with no usable Firebase app and no explicit AUTH_DISABLED,
    the endpoint is unavailable rather than open.
    """
    if AUTH_DISABLED:
        # Local development only (see the docstring): with auth disabled there
        # is no identity at all, so the compliance decision endpoints — which
        # read a role from the claims and refuse anyone without one — would
        # 403 the console's own demo in an environment that is already fully
        # open. The stand-in claims therefore carry the decider role. The
        # production path below still reads the role from the verified token.
        return {"uid": "auth-disabled", "email": "", "role": "mlro"}

    if not _FIREBASE_READY:
        reason = _FIREBASE_ERROR or "no application default credentials"
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication is unavailable: Firebase Admin credentials could "
                f"not be initialised ({reason}). Set AUTH_DISABLED=true only for "
                "local development."
            ),
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid authorization header"
        )

    token = authorization.replace("Bearer ", "", 1)
    try:
        return auth.verify_id_token(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


# ==========================================================================
# HELPERS
# ==========================================================================

def _extract_form(payload: Any) -> Dict[str, Any]:
    """Accept either {"form": {...}} or a bare form object.

    The wizard sends the wrapper; accepting a bare object keeps curl and the
    tests honest about the actual field set.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="A JSON object body is required")
    if isinstance(payload.get("form"), dict):
        return payload["form"]
    return payload


_AGENT: Optional[VendorQualificationAgent] = None


def get_agent() -> VendorQualificationAgent:
    """One agent per process.

    Constructing the agent initialises the Vertex AI client and the corpus
    handle, which is not something to redo on every request. It is also the
    seam the API tests replace, so a test can run the routes without Vertex.
    """
    global _AGENT
    if _AGENT is None:
        _AGENT = VendorQualificationAgent()
    return _AGENT


def _audit(dossier: Dict[str, Any], context: str) -> Dict[str, Any]:
    result = contract.audit(dossier)
    if not result["valid"]:
        print(
            f"[contract] {context}: {len(result['violations'])} violation(s) on a "
            f"{dossier.get('qualification_status')} dossier — "
            f"guardrail_check_passed downgraded to False"
        )
        for violation in result["violations"]:
            print(f"[contract]   - {violation}")
    return result


INCOMPLETE_MESSAGE = "Fill mandatory fields to compute risk score."


def _deterministic_preview(form: Dict[str, Any]) -> Dict[str, Any]:
    """The score-only view used by POST /score. No LLM, no RAG.

    This is what the wizard's live gauge renders, so it must stay available
    when Vertex AI is not: the number it shows is computed locally.

    Guardrail: an incomplete submission is not assessed. Every factor falls back
    to its neutral anchor, so the weighted average lands on 1.90 — a real
    arithmetic result built entirely from defaults, which reads on screen as a
    Medium Risk verdict for a form nobody has filled in. That contradicts
    "never assume missing data" and the contract rule that REJECTED_INCOMPLETE
    carries no score, so the incomplete case returns nulls plus the missing
    list. The factor table still travels, because showing which factors are
    unassessed is what tells the officer what to supply.
    """
    report = validation_lib.validate(form)
    if not report.is_complete:
        return {
            "qualification_status": "REJECTED_INCOMPLETE",
            "guardrail_message": INCOMPLETE_MESSAGE,
            "is_complete": False,
            "missing_count": len(report.missing),
            "missing_fields": [m.to_dict() for m in report.missing],
            "advisory_count": len(report.advisory),
            "advisory_fields": [m.to_dict() for m in report.advisory],
            "weighted_risk_score": None,
            "base_score": None,
            "assigned_risk_tier": None,
            "base_tier": None,
            "due_diligence_level": "",
            "refresh_cycle": "",
            "overridden": False,
            "factors": [],
            "unknown_factors": [],
            "triggers": [],
            "trigger_reasons": [],
            "derivation_notes": [],
        }

    signal_bag = signal_lib.build_signals(form)
    assessment = risk_engine.assess(signal_bag)
    override = trigger_lib.apply_mandatory_edd(assessment, signal_bag)

    if override.triggered and override.score_override is not None:
        score = float(override.score_override)
    else:
        score = float(assessment.weighted_risk_score)
    tier = override.tier_override if override.triggered else assessment.assigned_risk_tier
    # DD level and refresh cycle follow the final score, not the base score.
    # Deriving them from the base would render a gauge reading "High Risk (EDD)"
    # beside "Standard DD, refresh in 36 months" for a vendor escalated only by
    # a trigger.
    final_tier, dd_level, cycle = risk_engine.tier_for(Decimal(str(score)))
    if override.triggered and tier:
        final_tier = tier

    return {
        "qualification_status": "ASSESSED",
        "guardrail_message": "",
        "is_complete": True,
        "missing_count": len(report.missing),
        "missing_fields": [m.to_dict() for m in report.missing],
        "advisory_count": len(report.advisory),
        "weighted_risk_score": round(score, 2),
        "base_score": float(assessment.weighted_risk_score),
        "assigned_risk_tier": final_tier,
        "base_tier": assessment.assigned_risk_tier,
        "due_diligence_level": dd_level,
        "refresh_cycle": cycle,
        "overridden": override.triggered,
        "factors": [f.to_dict() for f in assessment.factors],
        "unknown_factors": assessment.unknown_factors,
        "triggers": [t.to_dict() for t in override.triggers],
        "trigger_reasons": trigger_lib.trigger_reasons(override.triggers),
        "derivation_notes": signal_bag.get("derivation_notes") or [],
    }


# ==========================================================================
# HEALTH + MATRIX
# ==========================================================================

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check. Deliberately unauthenticated and dependency-free: it must
    answer even when Vertex AI or Firebase is unavailable, so it reports the
    state of those dependencies rather than failing on them."""
    return {
        "status": "healthy",
        "service": "vendor-qualification-backend",
        "tool": "tool_1_vendor_qualification_and_risk_tiering",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "project_id": PROJECT_ID,
            "firebase_project_id": FIREBASE_PROJECT_ID,
            "location": LOCATION,
            "rag_corpus_configured": bool(RAG_CORPUS),
            "rag_corpus_shared_with_contract_review": True,
            "auth_enabled": not AUTH_DISABLED,
            "firebase_ready": _FIREBASE_READY,
            "schema_validation_available": contract.json_schema_available(),
        },
        "policy": {
            "document": POLICY_DOCUMENT,
            "form": FORM_DOCUMENT,
            "score_range": [1.0, 3.0],
        },
    }


@app.get("/api/vendors/qualifications/matrix")
async def matrix_endpoint() -> Dict[str, Any]:
    """The full deterministic matrix, so the UI can document the scoring
    without hardcoding it: factor weights and anchors, tier bands, the
    mandatory trigger catalogue, and the Appendix F weights and gate.

    Unauthenticated: it contains no vendor data, only the policy constants
    that are already in the repository.
    """
    return {
        "policy_document": POLICY_DOCUMENT,
        "form_document": FORM_DOCUMENT,
        "score_range": {"min": 1.0, "max": 3.0},
        "tier_bands": [
            {
                "upper_bound": float(upper),
                "tier": label,
                "due_diligence_level": dd_level,
                "refresh_cycle": cycle,
            }
            for upper, label, dd_level, cycle in risk_engine.TIER_BANDS
        ],
        "factors": risk_engine.factor_catalog(),
        "triggers": [
            {
                "code": t.code,
                "label": t.label,
                "policy_section": t.policy_section,
                "policy_basis": t.policy_basis,
                "minimum_controls": t.minimum_controls,
            }
            for t in trigger_lib.MANDATORY_EDD_TRIGGERS
        ],
            "mandatory_edd_floor": float(trigger_lib.FORCED_EDD_FLOOR),
        "strategic_spend_threshold_aed": risk_engine.SPEND_STRATEGIC_THRESHOLD_AED,
        "appendix_f": {
            "weights": dict(vq_schemas.APPENDIX_F_WEIGHTS),
            "pass_threshold": vq_schemas.APPENDIX_F_PASS_THRESHOLD,
            "category_minimum_ratio": vq_schemas.APPENDIX_F_CATEGORY_MINIMUM_RATIO,
        },
        "statuses": list(vq_schemas.QUALIFICATION_STATUSES),
        "tiers": list(vq_schemas.RISK_TIERS),
    }


# ==========================================================================
# AUTO-FILL — no LLM
# ==========================================================================

@app.post("/api/vendors/qualifications/auto-fill")
async def auto_fill_endpoint(
    file: UploadFile = File(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Read a Trade Licence or Company Profile and propose pre-filled form fields.

    Accepts a PDF or a .docx up to 50 MB. Never calls a model: the mapping is a
    label table in `company_profile`, so a licence is parsed the same way every
    time and a reviewer can see which line each value came from.

    A file that is readable but holds none of the expected labels answers 200
    with `extracted: true` and an empty `fields` list, which is different from a
    scan (200, `extracted: false`) and from a bad file (415 or 413). The wizard
    tells those three apart.

    This proposes; it does not decide. Only fields the caller sends back are
    written to the form, and the wizard fills only the ones that are empty, so an
    officer who has already typed something keeps it.
    """
    data = await file.read()
    try:
        result = extraction_lib.extract_text(data, file.filename or "")
    except extraction_lib.UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except extraction_lib.UnsupportedUpload as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    if not result.extracted:
        # A clean 200 with a reason. The wizard shows it and leaves the form
        # exactly as it was — a scanned licence is a normal thing to upload, not
        # a request error.
        return {
            "status": "success",
            "data": {
                "extracted": False,
                "reason": result.reason,
                "fields": [],
                "pages": result.pages,
                "warnings": result.warnings,
            },
        }

    fields = profile_lib.extract_company_fields(result.text)
    return {
        "status": "success",
        "data": {
            "extracted": True,
            "fields": [f.to_dict() for f in fields],
            "pages": result.pages,
            "chars_read": result.chars,
            "warnings": result.warnings,
            "filename": file.filename or "",
            "policy": {"document": POLICY_DOCUMENT, "form": FORM_DOCUMENT},
        },
    }


# ==========================================================================
# VALIDATE — no LLM
# ==========================================================================

@app.post("/api/vendors/qualifications/validate")
async def validate_endpoint(
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Run the NH-PQF-001 completeness gate and return the missing-data request.

    Never calls a model. The wizard calls this on submit to show the officer
    what is outstanding before spending anything on an assessment.
    """
    form = _extract_form(payload)
    report = validation_lib.validate(form)
    return {
        "status": "success",
        "data": {
            "report": report.to_dict(),
            "missing_data_request": validation_lib.missing_data_request(report),
            "policy": {
                "document": POLICY_DOCUMENT,
                "form": FORM_DOCUMENT,
            },
        },
    }


# ==========================================================================
# SCORE — no LLM
# ==========================================================================

@app.post("/api/vendors/qualifications/score")
async def score_endpoint(
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Deterministic weighted risk score, tier, factors and triggers.

    No LLM, no RAG: the score is the same value the full dossier reports, and
    it is available even when the corpus is down.

    An incomplete submission returns REJECTED_INCOMPLETE with no score or tier,
    per the guardrail. The missing-data request travels with it so the wizard
    can show the officer what is still outstanding.
    """
    form = _extract_form(payload)
    return {"status": "success", "data": _deterministic_preview(form)}


# ==========================================================================
# GUIDANCE — RAG policy citations, no score
# ==========================================================================

@app.post("/api/vendors/qualifications/guidance")
async def guidance_endpoint(
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Policy text retrieved for the inputs so far. No score, no narrative.

    This backs the wizard's Live Policy Guidance panel, so the officer can see
    which rule a later answer will engage *while* filling the form rather than
    only after submitting it. It is deliberately separate from /score, which
    stays pure and offline: retrieval needs Vertex AI, and a live gauge that
    failed whenever the corpus was down would be useless.

    Incomplete forms are allowed here, unlike /score. Showing policy text for a
    partially filled form is the point — the officer is still deciding, and the
    retrieved rules explain what the remaining answers will trigger.
    """
    form = _extract_form(payload)
    report = validation_lib.validate(form)
    agent = get_agent()

    try:
        # retrieve_policy_facts is synchronous and blocks on the Vertex AI
        # call, so it runs off the event loop.
        context = await run_in_threadpool(agent.retrieve_policy_facts, form)
    except Exception as exc:  # noqa: BLE001
        # Guidance is advisory. A corpus outage must not break the form.
        return {
            "status": "success",
            "data": {
                "available": False,
                "reason": f"Policy retrieval unavailable: {exc}",
                "citations": [],
                "rules": [],
                "jurisdiction_tier": None,
                "document": POLICY_DOCUMENT,
            },
        }

    retrieval = context.get("retrieval") or {}
    citations = [
        c.to_dict() for c in (context.get("citations") or [])
        if getattr(c, "quote", "") or getattr(c, "rule_applied", "")
    ]

    return {
        "status": "success",
        "data": {
            "available": bool(retrieval or citations),
            "reason": "",
            "is_complete": report.is_complete,
            "document": POLICY_DOCUMENT,
            "jurisdiction_tier": retrieval.get("jurisdiction_tier"),
            "category": retrieval.get("category"),
            "rules": retrieval.get("rules") or [],
            "citations": citations,
            "cited_count": sum(1 for c in citations if c.get("verified")),
            "total_count": len(citations),
        },
    }


# ==========================================================================
# QUALIFY — full dossier
# ==========================================================================

@app.post("/api/vendors/qualifications/prequalification")
async def prequalification_endpoint(
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """The Appendix F 0-100 prequalification sheet, separate from the 1.00-3.00
    risk score. Model: gemini-2.5-pro, the same narrative pass the dossier uses.

    Distinct from /score on purpose. The weighted risk score is deterministic and
    answers instantly; Appendix F is a judgement over the three categories the
    supplier asserts about itself, so it needs the model and cannot be previewed
    as a live gauge. Both are returned by /qualify, but the wizard needs them
    separately while the officer is still editing.

    Requires a complete form: scoring a half-filled prequalification sheet would
    be the same assumption the risk-score guardrail refuses.
    """
    form = _extract_form(payload)
    report = validation_lib.validate(form)
    if not report.is_complete:
        return {
            "status": "success",
            "data": {
                "available": False,
                "assessed": False,
                "reason": "Fill mandatory fields to compute the prequalification score.",
                "missing_count": len(report.missing),
            },
        }

    agent = get_agent()
    signal_bag = signal_lib.build_signals(form)
    assessment = risk_engine.assess(signal_bag)
    override = trigger_lib.apply_mandatory_edd(assessment, signal_bag)

    narrative = await agent._generate_narrative(
        form, signal_bag, assessment, override,
    )
    raw = (narrative or {}).get("appendix_f_assessment") or {}
    sheet = vq_schemas.appendix_f_scores(raw)
    sheet["assessed"] = bool(raw)

    return {
        "status": "success",
        "data": {
            "available": True,
            "reason": "",
            "model": agent.model_name,
            "missing_count": 0,
            **sheet,
        },
    }


@app.post("/api/vendors/qualifications/qualify")
async def qualify_endpoint(
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Run the full qualification and return the output_schema.md dossier.

    Two RAG passes (retrieval on flash, narrative on pro) around a
    deterministic core. An incomplete submission short-circuits to
    REJECTED_INCOMPLETE before either pass.
    """
    form = _extract_form(payload)
    dossier = await get_agent().qualify(form)
    audit = _audit(dossier, "qualify")
    # The audit travels inside the dossier as well as beside it, so a client
    # that persists `data` keeps the evidence that the dossier was checked.
    dossier = dict(dossier, contract_audit=audit)
    return {
        "status": "success",
        "data": dossier,
        "contract": {
            "valid": audit["valid"],
            "schema_checked": audit["schema_checked"],
            "violations": audit["violations"],
        },
    }


# ==========================================================================
# STREAM — SSE
# ==========================================================================

def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@app.post("/api/vendors/qualifications/stream")
async def stream_endpoint(
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> StreamingResponse:
    """Stream the qualification as it happens (SSE).

    Event types, in order:
        stage        a labelled step began
        validation   the completeness verdict
        jurisdiction the country tier established by retrieval
        category     the vendor category retrieved from the policy matrix
        retrieval    one policy rule was retrieved and cited
        score        the deterministic weighted score, tier and factor table
        triggers     one mandatory EDD trigger fired
        dossier      the final output_schema.md payload
        error        something failed; the dossier still follows
    """
    form = _extract_form(payload)

    async def sse_gen():
        dossier: Optional[Dict[str, Any]] = None
        try:
            async for event in get_agent().qualify_stream(form):
                if event.get("type") == "dossier":
                    dossier = event.get("dossier")
                yield _sse(event)
            if isinstance(dossier, dict):
                # The audit always travels with the final dossier, so the UI can
                # show whether the server verified the payload it just sent.
                audit = _audit(dossier, "stream")
                yield _sse({
                    "type": "dossier",
                    "dossier": dict(dossier, contract_audit=audit),
                })
        except Exception as exc:  # noqa: BLE001
            yield _sse({
                "type": "error",
                "message": f"Qualification failed: {exc}",
            })

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ==========================================================================
# SUPPLIER PORTAL
#
# Reached by email link, so there is no Firebase token: the path token is the
# only credential. Everything a supplier can read goes through
# `supplier_portal.supplier_view`, which copies an allowlist of fields rather
# than filtering the dossier — see that function for why it is built that way.
#
#     GET  /api/supplier/{token}                   the supplier's own view
#     PUT  /api/supplier/{token}/form/{section}    save one form section
#     POST /api/supplier/{token}/documents         upload + extract
#     GET  /api/approvals/{token}                  render a brief ONLY
#     POST /api/approvals/{token}                  record a decision
#     POST /api/compliance/alerts/{id}/decision    MLRO only
# ==========================================================================


@app.get("/api/supplier/{token}")
async def supplier_portal_view(token: str) -> Dict[str, Any]:
    """What the supplier sees. No login, and no internal risk output.

    Deliberately takes no `Depends(verify_firebase_token)`: requiring a login
    would defeat the purpose of an emailed link. The token is the credential.
    """
    session = supplier_portal._load_session(token)
    return supplier_portal.supplier_view(session)


@app.put("/api/supplier/{token}/form/{section}")
async def supplier_portal_save_section(
    token: str, section: str, payload: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Save one section of the supplier's own form.

    `section` must be in SUPPLIER_SECTIONS. An unknown section is a 404 rather
    than a 403: confirming which internal sections exist would itself be a
    disclosure.
    """
    session = supplier_portal._load_session(token)
    if section not in supplier_portal.SUPPLIER_SECTIONS:
        raise HTTPException(status_code=404, detail="Unknown section.")

    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise HTTPException(status_code=400, detail="`fields` must be an object.")

    # Only declared, supplier-writable fields are stored. A caller cannot use
    # this route to plant a value for a field outside its own form.
    for name, value in fields.items():
        if name not in supplier_portal.SUPPLIER_VISIBLE_FORM_FIELDS:
            raise HTTPException(
                status_code=400, detail=f"Unknown field: {name}"
            )
        session.form[name] = value

    # Any change to the submission invalidates a computed report. Without this
    # a cached REJECTED_INCOMPLETE — or worse, a cached tier that no longer
    # matches the answers — would survive the edit that was supposed to
    # resolve it.
    supplier_portal.clear_dossier(token)
    supplier_portal._save_session(session)
    return {
        "saved": sorted(fields),
        "status": session.neutral_status,
        "clarifications": [
            {"field": c.field, "question": c.question} for c in session.clarifications
        ],
    }


@app.post("/api/supplier/{token}/validate")
async def supplier_portal_validate(token: str) -> Dict[str, Any]:
    """The completeness gate over what the server actually holds. No model.

    The wizard calls this after saving the sections and before `/submit`, so a
    supplier sees the same missing list the buyer's gate would produce instead
    of submitting blind and leaving an unscored "Incomplete submission" on the
    buyer's dashboard with no way to fix it from their side. Token-gated like
    every other supplier route; the report carries form labels and reasons only
    — no score, no tier, nothing internal.
    """
    session = supplier_portal._load_session(token)
    # The gate must see what the buyer's gate sees: `session.form` alone
    # misses the documents rule, whose evidence lives in `session.documents`
    # rather than in the form. `qualification_payload` is the same assembly
    # the report pipeline validates, so the two lists can never diverge.
    report = validation_lib.validate(
        supplier_portal.qualification_payload(session))
    return {"report": report.to_dict()}


@app.post("/api/supplier/{token}/documents")
async def supplier_portal_upload_document(
    token: str,
    file: UploadFile = File(...),
    document_code: str = Form(""),
) -> Dict[str, Any]:
    """Accept a supplier document and run it through extraction.

    A document the supplier uploads is compared with what they typed. Where the
    two disagree the supplier is asked a neutral question; nothing about risk,
    screening or tier is returned, in any branch, including the failure paths.
    """
    session = supplier_portal._load_session(token)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The file was empty.")
    try:
        record = await run_in_threadpool(
            supplier_portal.store_document,
            session,
            file.filename or "upload",
            data,
            document_code,
        )
    except extraction_lib.UploadTooLarge:
        raise HTTPException(status_code=413, detail="That file is larger than 50 MB.")
    except extraction_lib.UnsupportedUpload:
        raise HTTPException(
            status_code=415,
            detail="Upload a PDF or a .docx file.",
        )

    # State machine sync. A document uploaded against a sent-back submission
    # is the supplier acting on the rejection: extraction above already ran
    # the re-check (it reconciles the new file with the form and raises
    # clarifications where they disagree), so the vendor moves back to the
    # buyer's queue — ACTION_REQUIRED → UNDER_REVIEW in the status machine's
    # vocabulary — instead of sitting in "Action required" with nobody watching.
    recheck: str | None = None
    if (
        session.submitted_at is not None
        and session.neutral_status == "Action required"
    ):
        session.neutral_status = "Under review"
        supplier_portal._save_session(session)
        recheck = (
            "Document re-check: corrected upload received — status moved "
            "from Action required to Under review."
        )
        try:
            log_notification(
                "recheck",
                f"Document re-check: {session.vendor_name or 'A vendor'} — back under review",
                session.invited_by or "",
                False,
                recheck,
                token=token,
                link=supplier_portal.report_link(token),
            )
        except Exception:  # noqa: BLE001 - the status move is the contract
            pass

    return {
        # Metadata only: the stored bytes are the preview route's to serve,
        # and no upload response should carry a base64 blob (or trip the
        # neutral-vocabulary firewall with encoded file content).
        "document": supplier_portal.without_preview(record),
        "status": session.neutral_status,
        "recheck": recheck,
        "clarifications": [
            {"field": c.field, "question": c.question} for c in session.clarifications
        ],
    }


# `api_route` (not the `get` decorator) so HEAD is declared explicitly: the
# vendor file's preview probe sends HEAD to learn whether stored bytes exist,
# and FastAPI only advertises the methods you list — `allow: GET` would turn
# every probe into a 405 and every document into "No preview".
@app.api_route(
    "/api/supplier/{token}/documents/{reference}", methods=["GET", "HEAD"]
)
def supplier_portal_get_document(token: str, reference: str) -> Response:
    """Serve the stored bytes of one upload, for the buyer's preview link.

    The token in the path is the credential exactly as on the upload route:
    whoever may file the document may read it back. What this returns is the
    supplier's own file — never an extracted field, never a status about risk —
    and documents that were stored without a preview copy (too large, or seeded
    before previews existed) are a plain 404 rather than an error the caller has
    to decode.
    """
    session = supplier_portal._load_session(token)
    record = supplier_portal.find_document(session, reference)
    if record is None or not record.get("content"):
        raise HTTPException(status_code=404, detail="No stored copy of that document.")

    # An ASCII stand-in for the filename: Content-Disposition cannot carry the
    # original's arbitrary characters quoted, and the browser shows this name
    # only when it will not render the file inline.
    safe_name = re.sub(r"[^\w.\- ]+", "_", reference) or "document"
    return Response(
        content=base64.b64decode(record["content"]),
        media_type=record.get("content_type") or "application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="{safe_name}"',
            "Cache-Control": "private, max-age=60",
        },
    )


@app.get("/api/approvals/{token}")
async def approval_brief(token: str) -> Dict[str, Any]:
    """Render an approval brief. Read-only by construction.

    Corporate mail scanners and link previewers GET every URL in an inbox. If
    GET recorded a decision, opening the mail would approve the vendor. This is
    the GET half of that pair; `approval_decision` is the POST half.
    """
    return supplier_portal.read_approval_brief(token)


@app.post("/api/approvals/{token}")
async def approval_decision(
    token: str,
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Record an approval decision. Requires an authenticated approver."""
    return supplier_portal.record_approval_decision(
        token,
        str(payload.get("decision") or ""),
        str(decoded.get("uid") or decoded.get("email") or ""),
    )


@app.post("/api/compliance/alerts/{alert_id}/decision")
async def compliance_alert_decision(
    alert_id: str,
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Record an MLRO's decision on a compliance alert.

    Buyer and GCEO roles are refused with 403. The role is read from the
    verified token's claims — never from a request header, which a caller can
    set. See `supplier_portal._require_mlro`.
    """
    return supplier_portal.decide_compliance_alert(
        alert_id,
        str(payload.get("decision") or ""),
        decoded,
        str(payload.get("justification") or ""),
        bool(payload.get("assisted")),
    )


@app.post("/api/vendors/qualifications/invitations")
async def supplier_invitation(
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Mint a supplier onboarding link. Internal, and therefore authenticated.

    There is no other way to obtain a token, which is the point: an unissued
    token resolves to the same 404 as an expired one, so the portal has no
    enumerable surface. A caller that does not know a vendor's name cannot
    discover whether an invitation already exists.

    Logging to the presenter outbox happens inside `supplier_portal.send_invitation`,
    so this route does not duplicate it.
    """
    vendor_name = str(payload.get("vendor_name") or "").strip()
    if not vendor_name:
        raise HTTPException(status_code=400, detail="vendor_name is required.")

    # Required, because the portal cannot do its job without it. An invitation
    # with no address is a form the supplier can fill in and a reviewer can
    # approve, with no way to tell either of them what happened — and on a
    # rejection that is the supplier sitting on a broken application indefinitely
    # while the buyer believes they were told to fix it.
    supplier_email = str(payload.get("supplier_email") or "").strip()
    if not supplier_email:
        raise HTTPException(
            status_code=400,
            detail=(
                "supplier_email is required. It is the only address the supplier "
                "can be reached at for an approval, a rejection, or a request for "
                "changes."
            ),
        )
    if not supplier_portal.valid_email(supplier_email):
        raise HTTPException(
            status_code=400, detail="supplier_email is not a valid address."
        )

    # The inviter is recorded on the session so a submission has someone to
    # notify. Taking it from the verified claims rather than the request body
    # means the notification cannot be aimed at an address the caller chose.
    inviter = str(decoded.get("email") or "").strip()
    token = supplier_portal.issue_token(
        vendor_name, invited_by=inviter, supplier_email=supplier_email
    )
    invitation = supplier_portal.send_invitation(token)
    supplier_portal.log_agent_activity(
        "Intake and Triage",
        f"Invitation emailed to {vendor_name} with a secure link.",
        rule="Spec 3.4 day 0",
        vendor=vendor_name,
        level="L3",
    )
    return {
        "vendor_name": vendor_name,
        "supplier_email": supplier_email,
        "token": token,
        "link": supplier_portal.invitation_link(token),
        "expires_in_days": supplier_portal.TOKEN_TTL_SECONDS // 86400,
        # Surfaced so the buyer can see, before sending the link, whether a
        # notification would actually reach anyone.
        "notifies": bool(inviter) or bool(os.environ.get("NOTIFICATION_TO", "").strip()),
        "smtp_configured": email_service.configured(),
        "invitation": invitation,
    }


@app.post("/api/supplier/{token}/submit")
async def supplier_submit(token: str) -> Dict[str, Any]:
    """Final submission: score it, notify the buyer, close it to further edits.

    Unauthenticated on purpose — the token is the credential, exactly as on
    every other supplier route. There is nothing here a stranger could do that
    the link they were already emailed could not: they can submit their own form
    and trigger an email to a buyer, which is the feature.

    The response deliberately carries no tier or score. The supplier is told
    they are under review and nothing more; `assessment` is populated for the
    buyer's benefit and the UI must not render it.
    """
    result = supplier_portal.submit_session(token, notify=True)
    return {
        "vendor_name": result["vendor_name"],
        "status": result["status"],
        "submitted_at": result["submitted_at"],
        "already_submitted": result["already_submitted"],
    }


# ==========================================================================
# Buyer-side reads of the supplier portal
#
# Authenticated, and deliberately kept out of the `/api/supplier/{token}`
# namespace: those routes are reachable by anyone holding a link and must not
# expose a conclusion. These are the only routes that return a tier or score.
# ==========================================================================

@app.get("/api/portal/submissions")
async def portal_submissions(
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Every supplier session, for the buyer dashboard."""
    rows = supplier_portal.list_submissions()
    return {"submissions": rows, "count": len(rows)}


@app.get("/api/portal/submissions/{token}")
async def portal_submission_detail(
    token: str,
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """One submission: the declared form as sent, plus the assessment."""
    return supplier_portal.buyer_view(token)


@app.post("/api/portal/submissions/{token}/decision")
async def portal_submission_decision(
    token: str,
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Record an internal approve/reject against a submission.

    POST-only for the same reason the approval brief is: a GET that could decide
    would let a mail gateway or a link previewer approve a supplier. The
    approver comes from the verified token, never from the body, so the audit
    record cannot be attributed to whoever the caller typed.
    """
    record = supplier_portal.record_submission_decision(
        token,
        str(payload.get("decision") or ""),
        supplier_portal.approver_from(decoded),
        str(payload.get("justification") or ""),
    )
    return {"decision": record, "submission": supplier_portal.buyer_view(token)}


async def _report_payload(
    token: str, force: bool = False, *, compute: bool = True
) -> Dict[str, Any]:
    """Build (or return) the full qualification dossier for a submission.

    Deliberately not run at submit time. The pipeline below makes two RAG
    passes and takes seconds to tens of seconds; doing it inline would make the
    supplier wait on it to save a form, and would mean a submission that timed
    out mid-pipeline had no record at all. Instead the report page asks for it,
    shows a progress state while it runs, and the result is cached on the
    session so a refresh is instant.

    Shared by the buyer's report endpoint and the reviewer's invite link: both
    screens show the same file, and a reviewer handed a thinner view would
    decide on different evidence than the buyer saw.

    `force` is accepted for compatibility but no longer bypasses a saved
    dossier: the assessment runs **once per vendor**. A refresh, a second
    press of Run Qualification, or "View full report" all open the same
    saved file — that is the contract the buyer's screens rely on. What
    makes a *new* run happen is the dossier being invalidated: a fresh
    upload or a saved form section clears it, and the next request then
    recomputes. `compute=False` serves only what is already cached
    regardless: the GET half of the review link sits in an email, and a
    corporate mail scanner's prefetch must never spend two RAG passes.

    The factor arithmetic in `run_assessment` is the authoritative score and is
    never overwritten by this — the dossier's own score is carried through the
    same `risk_engine` chain, so the two agree, and the dossier supplies the
    narrative, Appendix F, citations and controls around that number.
    """
    session = supplier_portal._load_session(token)
    if session.submitted_at is None:
        raise HTTPException(
            status_code=409,
            detail="This supplier has not submitted yet, so there is nothing to report on.",
        )

    cached = supplier_portal.get_dossier(token)
    if cached:
        return {
            "dossier": cached,
            "cached": True,
            "submission": _submission_envelope(token, session, cached),
        }
    if not compute:
        return {
            "dossier": None,
            "cached": True,
            "submission": _submission_envelope(token, session, {}),
        }

    # The agent is imported here rather than at module scope: main imports
    # supplier_portal, so a top-level import in the other direction is circular.
    #
    # The pipeline makes outbound model and RAG calls, so it fails for reasons
    # that have nothing to do with the submission. Left unhandled those arrive at
    # the buyer as a bare "Internal Server Error" with no indication whether a
    # retry is worthwhile, and the partial dossier is thrown away even though a
    # retry usually succeeds.
    try:
        dossier = await get_agent().qualify(
            supplier_portal.qualification_payload(session)
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("qualification pipeline failed for %s", token)
        raise HTTPException(
            status_code=502,
            detail=(
                "The qualification pipeline could not be completed for this "
                f"submission ({type(exc).__name__}). Nothing has been recorded, "
                "so retrying is safe."
            ),
        ) from exc

    # Stored before it is returned, so a dossier that computes successfully but
    # fails to serialise is not recomputed on every page refresh.
    supplier_portal.set_dossier(token, dossier)
    return {
        "dossier": dossier,
        "cached": False,
        "submission": _submission_envelope(token, session, dossier),
    }


@app.post("/api/portal/submissions/{token}/report")
async def portal_submission_report(
    token: str,
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
    payload: Optional[Dict[str, Any]] = Body(None),
) -> Dict[str, Any]:
    """The buyer's dossier endpoint; see `_report_payload` for the contract."""
    return await _report_payload(
        token, force=bool((payload or {}).get("force"))
    )


def _submission_envelope(
    token: str, session: supplier_portal.Session, dossier: Dict[str, Any]
) -> Dict[str, Any]:
    """The dossier plus the provenance the report screen has to show.

    `RiskScanRecord` on the client extends the dossier with who ran it and when,
    and the canvas renders the submitted form beside the assessment. The
    dossier carries none of that — it is the pipeline's output, not the
    submission's — so the report page would otherwise have to invent it, and a
    report that claimed an author it did not have would be worse than one that
    admitted the provenance is unknown.

    `submitted_form` is the form as declared, not `qualification_payload`: the
    evidence records the payload assembles for the validator are a
    representation of what was uploaded, and the reviewer is owed what the
    supplier actually sent.
    """
    return {
        "id": token,
        "vendor_id": dossier.get("vendor_id") or "",
        "invited_by": session.invited_by,
        "submitted_at": session.submitted_at,
        "created_at": session.created_at,
        "computed_at": session.internal.get("dossier_computed_at"),
        "submitted_form": session.form,
    }



@app.get("/api/console/notifications")
async def list_notifications(
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """The presenter outbox: every outbound Tool 1 notification."""
    return {"notifications": list_notification_log()}




@app.get("/api/vendors/qualifications/invitations")
async def list_supplier_invitations(
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """List all minted supplier invitations and their state."""
    items = supplier_portal.list_invitations()
    return {"items": items}


@app.delete("/api/vendors/qualifications/invitations/{token}")
async def delete_supplier_invitation(
    token: str,
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Withdraw one invitation: the supplier link stops working, and the
    submission — documents, review invites, alert, outbox rows — goes with it.

    Internal like the mint, and authenticated the same way: a caller that can
    invite a supplier can take the invitation back.
    """
    result = supplier_portal.delete_invitation(token)
    delete_notifications([token])
    return {"deleted": True, **result}

@app.post("/api/portal/submissions/{token}/review-invite")
async def create_review_invite(
    token: str,
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    reviewer_name = str(payload.get("reviewer_name") or "").strip()
    reviewer_email = str(payload.get("reviewer_email") or "").strip()
    invited_by = str(decoded.get("email") or decoded.get("uid") or "").strip()
    review_token = supplier_portal.issue_review_invite(token, reviewer_name, reviewer_email, invited_by)
    link = supplier_portal.review_link(review_token)
    try:
        session = supplier_portal._load_session(token)
        vendor = session.vendor_name if session else "supplier"
        subject = f"Internal review invite: {vendor} submission"
        body = f"""Hello {reviewer_name},

You've been invited to review a supplier submission ({vendor}) internally.

Review link (no login required): {link}

This link expires in 7 days.

Regards,
{invited_by or 'Team'}"""

        result = await _send_email(reviewer_email, subject, body)
        await log_notification_async(
            "review_invite",
            subject,
            reviewer_email,
            result.sent,
            result.detail or "",
            token=review_token,
            link=link,
        )
    except Exception:
        pass
    return {"review_token": review_token, "link": link}


@app.get("/api/console/compliance/alerts")
async def list_compliance_alerts(
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Every open and decided compliance alert, for the compliance queue."""
    rows: list[dict[str, Any]] = []
    for alert_id, alert in list(supplier_portal._ALERTS.items()):
        rows.append({
            "id": alert_id,
            "vendor_name": alert.get("vendor_name") or "",
            "status": "decided" if alert.get("decided") else "open",
            # The MLRO cannot decide an alert they cannot read. The reason
            # carries the tier and the score, which is why this endpoint is
            # buyer-side and authenticated while /api/supplier/* never is.
            "reason": alert.get("reason") or "",
            "decision": alert.get("decision"),
            "justification": alert.get("justification"),
            "decided_by": alert.get("decided_by"),
            "decided_at": alert.get("decided_at"),
            "supplier_token": alert.get("supplier_token") or "",
            "created_at": alert.get("created_at"),
        })
    rows.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return {"alerts": rows}


@app.post("/api/console/compliance/alerts/{alert_id}/decision")
async def console_compliance_alert_decision(
    alert_id: str,
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """MLRO records a decision on a compliance alert. MLRO-only by role claim."""
    decision = str(payload.get("decision") or "").strip().lower()
    if decision not in ("escalated", "cleared", "approved", "rejected"):
        raise HTTPException(
            status_code=400,
            detail="Decision must be one of escalated, cleared, approved, rejected.",
        )
    justification = str(payload.get("justification") or "")
    assisted = bool(payload.get("assisted"))
    # `rag` marks text that came back from the co-pilot's RAG pipeline; with
    # `assisted` it colours the Glass Box marker (RAG Context Applied).
    rag = bool(payload.get("rag"))
    result = supplier_portal.decide_compliance_alert(
        alert_id, decision, decoded, justification, assisted, rag=rag
    )
    return {
        "id": alert_id,
        "decision": result.get("decision"),
        "justification": result.get("justification"),
        "decided_by": result.get("decided_by"),
    }


@app.post("/api/console/compliance/alerts/{alert_id}/ai-draft")
async def console_compliance_alert_ai_draft(
    alert_id: str,
    payload: Optional[Dict[str, Any]] = Body(None),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """The Agentic Decision Co-Pilot's RAG draft analysis for one alert.

    Decider-only — the same `_require_mlro` gate as the decision this feeds —
    and read-only: it returns text for the justification textarea and records
    nothing. The retrieval context (screening findings, extracted document
    vault, policy specification rules) is assembled server-side and the model
    drafts over it with the policy corpus bound as the retrieval tool; the
    response says which source served the text (`rag-llm` or the
    deterministic `record-fallback`). The human decides whether any of it is
    filed; the decision endpoint, with `assisted` and `rag` when the draft
    was adopted, is what the Glass Box logs.
    """
    decision = str((payload or {}).get("decision") or "cleared")
    # Threadpool: the RAG pass may wait seconds on generation — on the event
    # loop that would freeze every other request (health, queue, uploads).
    return await run_in_threadpool(
        supplier_portal.draft_compliance_analysis, alert_id, decoded, decision
    )


@app.get("/api/console/glassbox")
async def list_glass_box(
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """The Glass Box: append-only audit log behind the compliance queue.

    Every decision arrives here as one row — who took it, at what level,
    which rule it relied on, and whether the Compliance Agent assisted.
    Authenticated like the queue itself: the log is buyer-side and the
    supplier routes never reach it. Claims are checked here as well as in
    the dependency, so an absent identity is 401 rather than an empty
    audit trail served to nobody in particular.
    """
    if not decoded:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    return {"events": supplier_portal.glassbox_events()}


@app.get("/api/console/agent-activity")
async def list_agent_activity(
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """The Agent Activity stream: what the background sub-agents have done.

    Append-only like the Glass Box — Document Intelligence reading uploads,
    Supplier Concierge chasing silent suppliers, Screening closing queues —
    and seeded with the demo fixture when empty so the dashboard always has
    history to scroll. Every action the system records appends here in the
    same shape. Authenticated like the console itself; supplier routes never
    reach it.
    """
    if not decoded:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    return {"events": supplier_portal.agent_activity_events()}


# ==========================================================================
# Presenter demo controls
#
# The console's presenter toolbar drives these: seed Vendor A/B/C through the
# real invitation → assessment flow, reset what was seeded, and run the agent
# clock (pause / play / +1 day). The clock is virtual; chasers and the
# console's overdue gates both read it, so advancing a day moves the whole
# demo together. See demo.py for the recipes and the chaser rules.
# ==========================================================================

@app.get("/api/demo/clock")
async def demo_get_clock(
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """The agent clock: virtual now, real now, and whether it is running."""
    import demo
    return demo.clock_status()


@app.post("/api/demo/clock")
async def demo_control_clock(
    payload: Dict[str, Any] = Body(...),
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Pause, play or advance the agent clock. `+1 day` fires crossed chasers."""
    import demo

    action = str(payload.get("action") or "").strip().lower()
    if action == "pause":
        clock = demo.set_running(False)
        return {"clock": clock, "chasers": []}
    if action == "play":
        demo.set_running(True)
        # A clock left running while the tab was closed has crossed days
        # nobody watched: settle them before reporting.
        fired = demo.run_chasers()
        return {"clock": demo.clock_status(), "chasers": fired}
    if action in ("advance", "advance_day", "+1"):
        days = payload.get("days")
        try:
            days = int(days) if days is not None else 1
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="days must be an integer.")
        clock = demo.advance_days(days)
        fired = demo.run_chasers()
        return {"clock": clock, "chasers": fired}
    if action == "reset":
        return {"clock": demo.reset_clock(), "chasers": []}
    raise HTTPException(
        status_code=400,
        detail="action must be one of: pause, play, advance, reset.",
    )


@app.post("/api/demo/run-vendor/{which}")
async def demo_run_vendor(
    which: str,
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Seed Vendor A (SDD), B (CDD) or C (EDD) through the real flow."""
    import demo
    return demo.run_vendor(which, str(decoded.get("email") or "").strip())


@app.post("/api/demo/reset")
async def demo_reset(
    decoded: Dict[str, Any] = Depends(verify_firebase_token),
) -> Dict[str, Any]:
    """Drop every seeded vendor, their alerts, their outbox rows, and reset the clock."""
    import demo
    return demo.reset_demo()


@app.on_event("startup")
async def _start_agent_clock() -> None:
    """Run the chaser loop for the life of the server."""
    try:
        import demo
        demo.start()
    except Exception:  # noqa: BLE001 - the demo loop must not block boot
        pass


@app.on_event("shutdown")
async def _stop_agent_clock() -> None:
    try:
        import demo
        demo.stop()
    except Exception:  # noqa: BLE001
        pass


@app.get("/api/portal/review/{review_token}")
async def get_review_invite(review_token: str) -> Dict[str, Any]:
    """Invite metadata plus the submission and any cached dossier.

    `compute=False`: a GET never runs the pipeline. This link is quoted in an
    email, so mail scanners and link previewers fetch it before any human
    opens it, and a prefetch must not spend two RAG passes. The reviewer's own
    access (the POST to `/validate` below) is what computes a missing dossier.
    """
    invite = supplier_portal.get_review_invite(review_token)
    submission_token = invite["submission_token"]
    return {
        "invite": invite,
        "submission": supplier_portal.buyer_view(submission_token),
        "report": await _report_payload(submission_token, compute=False),
    }



@app.post("/api/portal/review/{review_token}/validate")
async def validate_review_invite(review_token: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Verify the reviewer against their invite and hydrate the full dossier.

    The name/email check runs before any dossier work, so a wrong pair never
    triggers the pipeline. The dossier is then the same payload the buyer's
    report page renders — vendor header, dual scores, submitted form, policy
    citations, provenance — computed on first access if the buyer has not run
    it yet, cached thereafter.
    """
    invite = supplier_portal.get_review_invite(review_token)
    reviewer_name = str(payload.get("reviewer_name") or "").strip()
    reviewer_email = str(payload.get("reviewer_email") or "").strip()
    # allow if match or if invite doesn't specify strict? but better validate
    inv_name = str(invite.get("reviewer_name") or "").strip().lower()
    inv_email = str(invite.get("reviewer_email") or "").strip().lower()
    if inv_name and reviewer_name.lower() != inv_name:
        raise HTTPException(status_code=403, detail="Reviewer name does not match invite")
    if inv_email and reviewer_email.lower() != inv_email:
        raise HTTPException(status_code=403, detail="Reviewer email does not match invite")
    submission_token = invite["submission_token"]
    return {
        "invite": invite,
        "submission": supplier_portal.buyer_view(submission_token),
        "report": await _report_payload(submission_token),
    }
