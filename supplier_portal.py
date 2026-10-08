"""
Supplier portal — token-addressed, login-free access to the onboarding wizard,
plus the internal approval and compliance-decision surfaces.

Why this file exists
--------------------
Until now every endpoint in this service assumed a signed-in buyer: the caller
arrives with a Firebase ID token proving they are staff. That assumption breaks
for the supplier, who by definition has no account. A supplier is reached by an
email link, so their only credential is the token in that link.

Two consequences drive the whole design:

1. **The token is the credential.** `GET /api/supplier/{token}` authorises on
   the token alone — no Firebase token, no login. A token is 256 bits from
   `secrets.token_urlsafe`, compared in constant time, and never logged.

2. **The supplier is not the reader of this system's risk output.** They are the
   *subject* of the assessment. Telling a supplier "you are High Risk (EDD)"
   before onboarding completes is tipping off: it tells them the answer the
   assessment has not finished reaching, and it hands them the reason, so they
   can address the reason. See `supplier_view()` for the allowlist that makes
   this structural rather than a matter of remembering to redact.

Tiers are deliberately absent from this module. `risk_engine` decides them; this
module only ever sees the neutral vocabulary in `NEUTRAL_STATUSES`.

NOTE — token store durability
-----------------------------
`_SESSIONS` is a process-local dict. It is correct for a single worker and for
the test suite, and it is *not* correct for a multi-replica deployment, where a
supplier whose upload lands on replica B would find their form missing on
replica A. Swap `_SESSIONS` for a shared store (Firestore or Redis) before
running more than one worker. It is the single seam: `_load_session` and
`_save_session` are the only two functions that touch it.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from extraction import extract_text

# --------------------------------------------------------------------------
# Neutral vocabulary
# --------------------------------------------------------------------------

# The only four words a supplier ever sees. Nothing in this module may widen
# this tuple — there is no code path that appends to it.
NEUTRAL_STATUSES = (
    "In progress",
    "Under review",
    "Action required",
    "Approved",
)

# The shortest rejection reason accepted. See `record_submission_decision`.
MIN_REJECTION_REASON = 10

# How long a link stays usable. Long enough for a supplier abroad to receive a
# Friday-afternoon email; short enough that a forwarded link has a shelf life.
TOKEN_TTL_SECONDS = 14 * 24 * 60 * 60

# Sections the supplier may write. Matches the wizard's five steps. Anything
# outside this set is a 404, not a 403 — we do not confirm the existence of
# internal-only sections.
SUPPLIER_SECTIONS = (
    "company",
    "business",
    "representative",
    "ownership",
    "references",
    "turnover",
    "banking",
    "disclosures",
    "products",
    "documents",
    "declaration",
)


@dataclass
class Clarification:
    """A neutral request for information shown to the supplier.

    `field` is a form field name, never an internal signal: the supplier may be
    told "your bank name does not match your bank letter", but never "your
    banking pattern is a STRATEGIC_SPEND trigger".
    """

    field: str
    question: str
    raised_at: float = field(default_factory=time.time)


@dataclass
class Session:
    token: str
    vendor_name: str = ""
    neutral_status: str = "In progress"
    form: Dict[str, Any] = field(default_factory=dict)
    documents: List[Dict[str, Any]] = field(default_factory=list)
    clarifications: List[Clarification] = field(default_factory=list)
    # The full internal dossier, if one has been computed. Never serialised to
    # the supplier; `supplier_view()` cannot reach it because it works from an
    # allowlist rather than from this object.
    internal: Dict[str, Any] = field(default_factory=dict)
    # Who minted the link. A submission has to notify somebody, and the
    # inviter is the only party we know is a real, internal, human mailbox.
    invited_by: str = ""
    # Where this supplier can be reached. Collected at invitation time and held
    # for the life of the submission, because an approval, a rejection or a
    # request for changes has to be deliverable. Without it the portal can email
    # the buyer about a submission and then go silent, and the only way a
    # supplier learns their form was sent back is to reopen the same link.
    supplier_email: str = ""
    # Set once, when the supplier submits. Until then the buyer cannot tell a
    # submitted form from an untouched one, because the neutral status is
    # deliberately identical for both.
    submitted_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) - self.created_at > TOKEN_TTL_SECONDS


_SESSIONS: Dict[str, Session] = {}
_SESSIONS_LOCK = threading.Lock()

# A hashed one-time approval token → decision. Mirrors the supplier token:
# GET renders a brief, only POST records a decision, so a mail-scanner
# prefetching links cannot approve anything.
_APPROVALS: Dict[str, Dict[str, Any]] = {}
_APPROVALS_LOCK = threading.Lock()

# Compliance alerts that need an MLRO decision.
_ALERTS: Dict[str, Dict[str, Any]] = {}
_ALERTS_LOCK = threading.Lock()

MLRO_ONLY_ROLES = frozenset({"mlro"})
# Compliance decisions are made by the MLRO and, standing beside them, the
# Subsidiary Compliance Officer — the same two roles the console's
# "Viewing as" selector offers the Inspect/Review queue and the clear/block
# decisions for. Any other role reads the queue but records nothing.
DECIDER_ROLES = MLRO_ONLY_ROLES | {"sco"}


# --------------------------------------------------------------------------
# Durability
#
# A supplier link is a 14-day artefact that gets emailed to a stranger. Holding
# it in a dict means a restart, a redeploy or a second worker silently
# invalidates links that a vendor is mid-way through filling in — and the
# failure looks like a phishing page rather than an outage. So the dict is a
# read cache and SQLite is the record of truth, written through on every save.
#
# Opt-in via SUPPLIER_PORTAL_DB. Unset means memory-only, which is what the
# tests use and what a single local uvicorn without the variable gets.
#
# This survives restarts and deploys but not multi-replica: each replica would
# still get its own file. That needs a shared backend (Firestore or Redis), and
# is the one thing here not yet sorted.
# --------------------------------------------------------------------------

_DB_PATH = os.environ.get("SUPPLIER_PORTAL_DB", "").strip()
_DB: Any = None
_DB_LOCK = threading.Lock()


def _db() -> Any:
    """Lazily open the store. Returns None when durability is not configured."""
    global _DB
    if not _DB_PATH:
        return None
    if _DB is None:
        import sqlite3

        _DB = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _DB.execute("PRAGMA journal_mode=WAL")
        _DB.execute(
            "CREATE TABLE IF NOT EXISTS portal_sessions ("
            "token TEXT PRIMARY KEY, created_at REAL NOT NULL, data TEXT NOT NULL)"
        )
        _DB.execute(
            "CREATE TABLE IF NOT EXISTS portal_items ("
            "kind TEXT NOT NULL, key TEXT NOT NULL, data TEXT NOT NULL,"
            "PRIMARY KEY (kind, key))"
        )
        _DB.commit()
        _restore()
    return _DB


def _session_row(session: Session) -> Dict[str, Any]:
    return {
        "token": session.token,
        "vendor_name": session.vendor_name,
        "neutral_status": session.neutral_status,
        "form": session.form,
        "documents": session.documents,
        "clarifications": [asdict(c) for c in session.clarifications],
        "internal": session.internal,
        "invited_by": session.invited_by,
        # Written here as well as read in `_session_row`'s callers. The reader
        # and the writer are separate functions, and a field added to only one of
        # them survives in memory until the next restart and is then gone — which
        # for the supplier's address means decisions silently stop being
        # deliverable after a deploy.
        "supplier_email": session.supplier_email,
        "submitted_at": session.submitted_at,
        "created_at": session.created_at,
        "last_seen_at": session.last_seen_at,
    }


def _persist_session(session: Session) -> None:
    db = _db()
    if db is None:
        return
    with _DB_LOCK:
        db.execute(
            "INSERT OR REPLACE INTO portal_sessions (token, created_at, data)"
            " VALUES (?, ?, ?)",
            (
                session.token,
                session.created_at,
                json.dumps(_session_row(session), default=str),
            ),
        )
        db.commit()


def _get_item(kind: str, key: str) -> Optional[Dict[str, Any]]:
    db = _db()
    if db is not None:
        with _DB_LOCK:
            cur = db.execute(
                "SELECT data FROM portal_items WHERE kind=? AND key=?",
                (kind, key),
            ).fetchone()
            if cur:
                try:
                    return {"payload": json.loads(cur[0])}
                except Exception:
                    return None
    return None


def _persist_item(kind: str, key: str, payload: Dict[str, Any]) -> None:
    db = _db()
    if db is None:
        return
    with _DB_LOCK:
        db.execute(
            "INSERT OR REPLACE INTO portal_items (kind, key, data) VALUES (?, ?, ?)",
            (kind, key, json.dumps(payload, default=str)),
        )
        db.commit()


def _get_item(kind: str, key: str) -> Optional[Dict[str, Any]]:
    db = _db()
    if db is not None:
        with _DB_LOCK:
            cur = db.execute(
                "SELECT data FROM portal_items WHERE kind=? AND key=?",
                (kind, key),
            ).fetchone()
            if cur:
                try:
                    return {"payload": json.loads(cur[0])}
                except Exception:
                    return None
    return None


def _restore() -> None:
    """Repopulate the caches from disk at startup, dropping expired rows.

    Expired sessions are deleted rather than loaded: a 14-day-old link must not
    come back to life on restart, and leaving them to accumulate would make the
    table grow without bound.
    """
    db = _db_unlocked()
    if db is None:
        return
    now = time.time()
    with _DB_LOCK:
        for token, created_at, blob in db.execute(
            "SELECT token, created_at, data FROM portal_sessions"
        ).fetchall():
            if now - created_at > TOKEN_TTL_SECONDS:
                continue
            row = json.loads(blob)
            _SESSIONS[token] = Session(
                token=row["token"],
                vendor_name=row.get("vendor_name", ""),
                neutral_status=row.get("neutral_status", "In progress"),
                form=row.get("form") or {},
                documents=row.get("documents") or [],
                clarifications=[Clarification(**c) for c in row.get("clarifications") or []],
                internal=row.get("internal") or {},
                invited_by=row.get("invited_by", ""),
                supplier_email=row.get("supplier_email", ""),
                submitted_at=row.get("submitted_at"),
                created_at=row.get("created_at", created_at),
                last_seen_at=row.get("last_seen_at", created_at),
            )
        glassbox_rows: List[Dict[str, Any]] = []
        activity_rows: List[Dict[str, Any]] = []
        for kind, key, blob in db.execute("SELECT kind, key, data FROM portal_items").fetchall():
            payload = json.loads(blob)
            if kind == "approval":
                _APPROVALS[key] = payload
            elif kind == "alert":
                _ALERTS[key] = payload
            elif kind == "glassbox":
                glassbox_rows.append(payload)
            elif kind == "agent_activity":
                activity_rows.append(payload)
        _merge_restored_rows(glassbox_rows, _GLASSBOX, _GLASSBOX_LOCK)
        _merge_restored_rows(activity_rows, _ACTIVITY, _ACTIVITY_LOCK)
        db.execute(
            "DELETE FROM portal_sessions WHERE created_at < ?",
            (now - TOKEN_TTL_SECONDS,),
        )
        db.commit()


def _get_item(kind: str, key: str) -> Optional[Dict[str, Any]]:
    db = _db()
    if db is not None:
        with _DB_LOCK:
            cur = db.execute(
                "SELECT data FROM portal_items WHERE kind=? AND key=?",
                (kind, key),
            ).fetchone()
            if cur:
                try:
                    return {"payload": json.loads(cur[0])}
                except Exception:
                    return None
    return None


def _db_unlocked() -> Any:
    """The connection without re-entering the creation path from `_restore`."""
    return _DB


def valid_email(address: str) -> bool:
    """Whether `address` is plausibly deliverable.

    Deliberately shallow. The only authority on whether a mailbox exists is
    delivering to it, and an over-strict regex rejects valid addresses
    (plus-addressing, some TLDs, long local parts) which is a worse failure than
    a bounce we report honestly.
    """
    candidate = (address or "").strip()
    if candidate.count("@") != 1:
        return False
    local, _, domain = candidate.partition("@")
    if not local or not domain or "." not in domain:
        return False
    if domain.startswith(".") or domain.endswith(".") or ".." in domain:
        return False
    return not any(ch.isspace() for ch in candidate)


def issue_token(
    vendor_name: str = "",
    invited_by: str = "",
    supplier_email: str = "",
) -> str:
    """Mint a supplier token and its session.

    `supplier_email` is the only route to telling this supplier anything after
    the invitation. It was not collected before, which left the portal able to
    email the buyer about a submission and then go silent — a rejection
    instructing the supplier to fix their form reached nobody, and the only way
    to learn it had happened was to reopen the same link by hand.
    """
    token = secrets.token_urlsafe(32)
    session = Session(
        token=token,
        vendor_name=vendor_name,
        invited_by=invited_by,
        supplier_email=(supplier_email or "").strip(),
    )
    with _SESSIONS_LOCK:
        _SESSIONS[token] = session
    _persist_session(session)
    return token


def send_invitation(token: str) -> Dict[str, Any]:
    """Email the supplier their onboarding link.

    Separate from `issue_token` so minting a token is not a network call, and so
    a failed send can be retried without minting a second token — a supplier who
    is sent two links has two sessions, and only one of them is the one the
    buyer is looking at.
    """
    session = _load_session(token)
    if not session.supplier_email:
        return {
            "sent": False,
            "detail": "No supplier email on file, so the invitation was not sent.",
            "recipient": "",
        }
    if not os.environ.get("SUPPLIER_PORTAL_BASE_URL"):
        return {
            "sent": False,
            "detail": "SUPPLIER_PORTAL_BASE_URL is unset, so no link could be built.",
            "recipient": session.supplier_email,
        }
    try:
        from services import email as email_service

        result = email_service.notify_supplier_invitation(
            session.supplier_email,
            session.vendor_name or "",
            invitation_link(token),
        )
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        return payload
    except Exception as exc:  # noqa: BLE001 - the invitation exists regardless
        return {
            "sent": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "recipient": session.supplier_email,
        }


def invitation_link(token: str) -> str:
    """The supplier-facing URL for a token.

    Kept beside `issue_token` so the path is written once. A token is only
    useful to a supplier as a link, and a link assembled at the call site is a
    link that can be assembled wrong.
    """
    return f"{_portal_base()}/supplier/onboarding/{token}"


def report_link(token: str) -> str:
    """The buyer-facing report URL for a token.

    This is what a submission notification must link to. It used to link to
    `invitation_link`, so "Review it here" dropped the reviewer onto the
    supplier's own form: a page they cannot submit, that shows them the
    questions the supplier still has to answer, and that contains no assessment
    at all. The report page authenticates them and renders the dossier.

    Same base URL as the supplier link — both are routes on the one web app —
    but a separate function rather than a flag, because the two audiences are
    the whole point and a caller should not be able to pick the wrong one by
    forgetting a boolean.
    """
    return f"{_portal_base()}/portal/submissions/{token}/report"


def _portal_base(required: bool = True) -> str:
    base = os.environ.get("SUPPLIER_PORTAL_BASE_URL", "").rstrip("/")
    if not base and required:
        raise HTTPException(
            status_code=503,
            detail=(
                "SUPPLIER_PORTAL_BASE_URL is not configured, so a link cannot "
                "be built."
            ),
        )
    if not base:
        base = os.environ.get("PORTAL_BASE_URL", "http://localhost:8000").rstrip("/")
    return base


def _load_session(token: str) -> Session:
    """Resolve a token, or fail indistinguishably.

    A missing token and an expired token both raise the same 404 with the same
    body. Distinguishing them tells an attacker whether a guessed token ever
    existed.
    """
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(token)
    if session is None:
        _reload_session(token)
        with _SESSIONS_LOCK:
            session = _SESSIONS.get(token)
    if session is None:
        raise HTTPException(status_code=404, detail="This link is not valid.")
    if session.is_expired():
        raise HTTPException(status_code=404, detail="This link is not valid.")
    session.last_seen_at = time.time()
    return session


def _save_session(session: Session) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS[session.token] = session
    _persist_session(session)


def _find_session(token: str) -> Optional[Session]:
    """Cache-then-disk lookup that returns None instead of raising.

    Used where a session is optional. `issue_token` and the approval flow share
    the same shape but different contracts: they must fail loudly on an unknown
    token, this must not.
    """
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(token)
    if session is not None:
        return session
    _reload_session(token)
    with _SESSIONS_LOCK:
        return _SESSIONS.get(token)


def _reload_session(token: str) -> None:
    """Re-read one session from disk into the cache."""
    db = _db()
    if db is None:
        return
    with _DB_LOCK:
        row = db.execute(
            "SELECT created_at, data FROM portal_sessions WHERE token = ?", (token,)
        ).fetchone()
    if row is None:
        return
    created_at, blob = row
    if time.time() - created_at > TOKEN_TTL_SECONDS:
        return
    data = json.loads(blob)
    session = Session(
        token=data["token"],
        vendor_name=data.get("vendor_name", ""),
        neutral_status=data.get("neutral_status", "In progress"),
        form=data.get("form") or {},
        documents=data.get("documents") or [],
        clarifications=[Clarification(**c) for c in data.get("clarifications") or []],
        internal=data.get("internal") or {},
        invited_by=data.get("invited_by", ""),
        supplier_email=data.get("supplier_email", ""),
        submitted_at=data.get("submitted_at"),
        created_at=data.get("created_at", created_at),
        last_seen_at=data.get("last_seen_at", created_at),
    )
    with _SESSIONS_LOCK:
        _SESSIONS.setdefault(token, session)


def reset_all() -> None:
    """Drop every session, approval, alert, Glass Box entry and Agent Activity
    event, on disk as well as in memory.

    Clearing only the dicts is not a reset: with SUPPLIER_PORTAL_DB set — which
    `main.py` triggers by loading .env — `_restore()` would repopulate them from
    the developer's real database on the next lookup, and any live session would
    quietly reappear.
    """
    # Open the store first. `_db()` runs `_restore()` on first use, so opening
    # it *after* the clears would repopulate the dicts from disk and the DELETE
    # would then wipe the rows out from under live objects.
    db = _db()
    with _SESSIONS_LOCK:
        _SESSIONS.clear()
    with _APPROVALS_LOCK:
        _APPROVALS.clear()
    with _ALERTS_LOCK:
        _ALERTS.clear()
    with _GLASSBOX_LOCK:
        _GLASSBOX.clear()
    with _ACTIVITY_LOCK:
        _ACTIVITY.clear()
    if db is not None:
        with _DB_LOCK:
            db.execute("DELETE FROM portal_sessions")
            db.execute("DELETE FROM portal_items")
            db.commit()


def _get_item(kind: str, key: str) -> Optional[Dict[str, Any]]:
    db = _db()
    if db is not None:
        with _DB_LOCK:
            cur = db.execute(
                "SELECT data FROM portal_items WHERE kind=? AND key=?",
                (kind, key),
            ).fetchone()
            if cur:
                try:
                    return {"payload": json.loads(cur[0])}
                except Exception:
                    return None
    return None


# --------------------------------------------------------------------------
# The tipping-off firewall
# --------------------------------------------------------------------------

# Fields a supplier may see on their own submission. An allowlist, not a
# denylist: a denylist has to be updated every time the dossier gains a field,
# and the field that is forgotten is the one that leaks. Here, adding an
# internal field to the dossier cannot expose it, because it was never on this
# list.
SUPPLIER_VISIBLE_FORM_FIELDS = (
    # company
    "legal_name", "registered_address", "country_of_incorporation",
    "date_of_incorporation", "year_of_commencement", "trade_license_no",
    "trade_license_expiry", "vat_registration_status", "vat_registration_no",
    "certifications",
    # business
    "nature_of_business", "goods_services_proposed", "geographic_coverage",
    "supply_type", "sensitivity", "products_services",
    # representative — identity, not their authority assessment
    "authorized_representative_name", "authorized_representative_designation",
    "authorized_representative_id", "authorized_representative_id_expiry",
    "authorized_representative_authority_basis", "compliance_contact",
    # ownership — what the supplier *declared*. `ubos` and the `pep_*` flags
    # were previously excluded on the reasoning that reflecting them back would
    # disclose the screening result. That was wrong in both directions: these
    # are the supplier's own declarations and the supplier is the only possible
    # source for them, so excluding the writes made the mandatory Item 34 UBO
    # declaration unsatisfiable and every portal submission came back
    # REJECTED_INCOMPLETE from the full pipeline. Reading a supplier's own
    # answer back to them discloses nothing; the firewall is the neutral status
    # vocabulary and the absence of any tier, score or screening outcome from
    # these responses, both of which are unaffected by what they may declare.
    "directors_and_owners", "ultimate_parent_company",
    "ownership_structure", "ownership_structure_chart",
    "bearer_shares", "nominee_shareholders",
    "ubos", "pep_present", "pep_family_member", "pep_close_associate",
    # references
    "client_references", "public_review_links",
    # turnover
    "turnover_year_1", "turnover_year_2", "turnover_year_3",
    "number_of_employees", "financial_statements_audited",
    # banking
    "bank_name_branch_country", "bank_account_name", "bank_account_number",
    "bank_iban", "swift_code", "payment_terms", "payment_structure",
    "third_party_payment", "foreign_account_payment", "cash_payment",
    "unusual_payment_explanation",
    # disclosures — the supplier's own declarations, not the risk read
    "address_change_declaration", "management_change_declaration",
    "intermediary_declaration", "subcontracting_declaration",
    "regulatory_history_declaration", "frequent_address_changes",
    "frequent_management_changes", "acts_as_intermediary",
    "uses_subcontractor", "prior_regulatory_matter",
    "engages_government_officials", "government_licensing",
    # declaration / signature
    "supplier_declaration", "authorized_signatory_name",
    "authorized_signatory_designation", "date_signed",
    "estimated_spend_aed", "single_contract_value_aed",
)

# Assertions for the tests: if the dossier grows one of these, the firewall is
# no longer sufficient and the test must fail loudly.
# `ubos` and the `pep_*` flags used to be listed here. They are the supplier's
# own declarations and are now writable and readable, because omitting them made
# the mandatory Item 34 UBO declaration impossible to satisfy and every portal
# submission failed the full pipeline as incomplete. What must never cross to
# the supplier is the *screening outcome* derived from them — which is what the
# rest of this set is.
NEVER_SUPPLIER_VISIBLE = frozenset({
    "weighted_risk_score", "assigned_risk_tier", "risk_tier",
    "mandatory_edd_triggered", "trigger_details", "risk_signals",
    "jurisdiction_assessment", "jurisdiction_tier", "missing_data",
    "sanctions", "sanctions_result", "screening_result", "edd_flags",
    "weighted_factors", "appendix_f", "citations", "risk_reasons",
    "guardrail_check_passed", "onboarding_flags", "next_steps",
})


def supplier_view(session: Session) -> Dict[str, Any]:
    """The only shape the supplier portal is allowed to emit.

    Built by copying named fields across, so the internal dossier on the
    session is never walked. `session.internal` is intentionally not read here;
    that is the point. A PEP or sanctions result sitting in `internal` cannot
    escape through this function because there is no code path from it to here.
    """
    visible = {
        name: session.form[name]
        for name in SUPPLIER_VISIBLE_FORM_FIELDS
        if name in session.form
    }
    leaked = NEVER_SUPPLIER_VISIBLE.intersection(visible)
    if leaked:
        # Would mean someone added an internal name to the allowlist above.
        raise RuntimeError(
            "supplier_view allowlist contains internal fields: "
            + ", ".join(sorted(leaked))
        )

    documents = [
        {
            "reference": doc["reference"],
            "document_code": doc.get("document_code", ""),
            "status": doc["status"],
            "extracted_fields": sorted(doc.get("extracted_fields", ())),
        }
        for doc in session.documents
    ]

    return {
        "vendor_name": session.vendor_name,
        # Neutral only. The internal qualification_status is not carried here
        # at all, so there is nothing to map at read time and nothing to leak
        # if a new internal status is introduced.
        "status": session.neutral_status,
        "statuses_you_may_see": list(NEUTRAL_STATUSES),
        "form": visible,
        "documents": documents,
        "clarifications": [
            {"field": c.field, "question": c.question} for c in session.clarifications
        ],
        # Their own outcome, and the reason the buyer gave with it.
        #
        # The rejection reason is deliberately included. A supplier told only
        # "Action required" cannot act on it: the clarifications they can see are
        # the ones document extraction raised, which are frequently not why the
        # submission was sent back. The reason is the reviewer's own words about
        # what to fix, and withholding it would leave them resubmitting the same
        # form. What it never contains is the tier, the score or any screening
        # result — those are not in the record, because `record_submission_decision`
        # does not copy them into the justification.
        "decision": _supplier_decision(session),
    }


def _supplier_decision(session: Session) -> Optional[Dict[str, Any]]:
    """The decision as the supplier is allowed to see it."""
    decision = session.internal.get("decision")
    if not decision:
        return None
    return {
        "decision": decision.get("decision"),
        "justification": decision.get("justification", ""),
        "decided_at": decision.get("decided_at"),
        # The tier and score are present on the internal record and are
        # deliberately not copied here. A reviewer may write "your turnover does
        # not support the spend" as a reason, which is about their form rather
        # than about its score.
        "can_resubmit": session.submitted_at is not None,
    }


# --------------------------------------------------------------------------
# Neutral mapping for internal outcomes
# --------------------------------------------------------------------------

def neutral_status_for(qualification_status: Optional[str]) -> str:
    """Collapse an internal outcome to one of the four supplier-visible words.

    Coarse on purpose. "Under review" covers conditional qualification and
    escalation alike, because the difference between them is precisely the
    commercial judgement the supplier must not receive.
    """
    status = (qualification_status or "").strip().upper()
    if status == "REJECTED_INCOMPLETE":
        return "Action required"
    if status in ("QUALIFIED", "APPROVED"):
        return "Approved"
    if status in ("CONDITIONALLY_QUALIFIED", "ESCALATED", "REFERRED"):
        return "Under review"
    return "In progress"


# --------------------------------------------------------------------------
# Document upload and extraction
# --------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Uploads up to this size keep a servable copy alongside their metadata, so the
# buyer's vendor file can preview the actual file (a trade licence, a passport
# copy, a bank letter). It is a convenience, not the archive: the session row
# is the wrong place to hide fifty megabytes, so anything larger keeps its
# metadata alone and the preview route honestly reports 404 for it.
MAX_DOCUMENT_PREVIEW_BYTES = 2 * 1024 * 1024

# Preview-only keys. `supplier_view` already builds its documents from named
# fields and never sees them; `buyer_view` and the upload response are given
# these keys to strip so no caller is handed a base64 blob by accident.
_PREVIEW_KEYS = ("content", "content_type")


def without_preview(document: Dict[str, Any]) -> Dict[str, Any]:
    """A document record with the preview payload removed — metadata only."""
    return {k: v for k, v in document.items() if k not in _PREVIEW_KEYS}


def document_filename(document: Dict[str, Any]) -> str:
    """The uploaded file's own name, for any list a human reads.

    New uploads record `filename` directly. Older records — seeded demos and
    uploads stored before the field existed — carry only a `reference` built
    as `{code}-{code}-{slug}.pdf` (seed) or `{code}-{name}.pdf` (live), so the
    name is recovered from there. A plain slug such as
    `bank-spend-confirmation.pdf` is split into words; anything that already
    reads as a real name passes through untouched.
    """
    name = str(document.get("filename") or "").strip()
    if name:
        return name
    reference = str(document.get("reference") or "").strip()
    code = str(document.get("document_code") or "").strip()
    candidate = reference
    if code:
        prefix = f"{code}-"
        # Seeds double the code (`62-62-...`); live uploads carry it once.
        while candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
    if not candidate:
        return reference
    slug = re.fullmatch(
        r"([a-z0-9]+(?:[-_][a-z0-9]+)+)\.(pdf|docx?|png|jpe?g)",
        candidate,
        re.IGNORECASE,
    )
    if slug:
        stem = re.sub(r"[-_]+", " ", slug.group(1))
        return f"{stem.title()}.{slug.group(2).lower()}"
    return candidate


def _attach_preview(record: Dict[str, Any], data: bytes, filename: str) -> None:
    """Keep the bytes on the record when they are small enough to serve back.

    The copy is what makes the buyer's "Preview" link real: without it the
    upload route extracts text and throws the file away, and there is nothing
    left to show. Stored base64 inside the session so it persists, restores
    and resets with the document it belongs to.
    """
    if len(data) > MAX_DOCUMENT_PREVIEW_BYTES:
        return
    record["content"] = base64.b64encode(data).decode("ascii")
    record["content_type"] = (
        mimetypes.guess_type(filename)[0] or "application/octet-stream"
    )


def find_document(session: Session, reference: str) -> Optional[Dict[str, Any]]:
    """The stored record for one reference on this session, if it exists."""
    for document in session.documents:
        if document.get("reference") == reference:
            return document
    return None

# Document code → the form field its extraction should confirm. Deliberately
# blunt: these are presence checks, not judgements.
_DOCUMENT_CODES = {
    "62": ("trade_license_no", "Trade licence"),
    "63": ("vat_registration_no", "VAT certificate"),
    "64": ("registered_address", "Company profile"),
    "68": ("bank_name_branch_country", "Bank letter"),
    "67": ("authorized_representative_id", "Emirates ID"),
}

# Questions the supplier is asked when a document disagrees with what they
# typed. The wording may name the document and the field — those are theirs.
# It may never name a risk, a tier, a screening result or a trigger: the
# supplier has to be able to answer the question without being told what it
# implies.
_CLARIFICATION_QUESTIONS = {
    "trade_license_no": "We could not read a licence number from your trade licence. Could you confirm the licence number and expiry date?",
    "vat_registration_no": "We could not read a TRN from your VAT certificate. Could you confirm the TRN?",
    "registered_address": "We could not read a registered address from your company profile. Could you confirm it?",
    "bank_name_branch_country": "The bank name on your bank letter does not match the one you entered. Which is correct?",
    "authorized_representative_id": "We could not read an Emirates ID or passport number. Could you confirm it?",
    "legal_name": "The company name on the document does not match the name you entered. Could you confirm your registered legal name?",
}

# Plain-English names for the fields we can ask about without echoing an
# internal key back at the supplier.
_FIELD_LABELS = {
    "legal_name": "company name",
    "trade_license_no": "trade licence number",
    "trade_license_expiry": "trade licence expiry date",
    "registered_address": "registered address",
    "country_of_incorporation": "country of incorporation",
    "date_of_incorporation": "date of incorporation",
    "year_of_commencement": "year of commencement",
    "vat_registration_no": "TRN",
    "compliance_contact": "compliance contact",
    "bank_name_branch_country": "bank name",
    "bank_account_name": "bank account name",
    "bank_account_number": "bank account number",
    "bank_iban": "bank IBAN",
}


def _clarification_question(field_name: str, from_document: Any) -> str:
    """A neutral question about one field.

    Two shapes, both non-disclosing: a mismatch names the field, and an
    unreadable field asks for it. Neither says why the field is being raised.
    """
    canned = _CLARIFICATION_QUESTIONS.get(field_name)
    if canned and "does not match" in canned:
        return canned
    label = _FIELD_LABELS.get(field_name, field_name.replace("_", " "))
    return (
        f"The {label} on the document does not match what you entered. "
        "Could you confirm which is correct?"
    )


def store_document(
    session: Session,
    filename: str,
    data: bytes,
    document_code: str = "",
) -> Dict[str, Any]:
    """Run a supplier upload through extraction and reconcile it with the form.

    Extraction failures become `clarifications` on the supplier session, never
    exceptions the supplier can read as an internal fault, and never a statement
    about risk.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That file is larger than 50 MB.")

    result = extract_text(data, filename)
    reference = f"{document_code}-{filename}" if document_code else filename

    if not result.extracted:
        # A scan, or a password-protected file. Ask for a readable copy. We do
        # not report which of the two it was.
        session.clarifications.append(Clarification(
            field="documents",
            question=(
                f"We could not read text from {reference}. Please upload a "
                "digital copy rather than a scan or photograph."
            ),
        ))
        record = {
            "reference": reference,
            "document_code": document_code,
            "filename": filename,
            "status": "Unreadable",
            "extracted_fields": [],
        }
        # A scan we could not read is still the supplier's file — the buyer's
        # vendor file may preview it even though it is not evidence.
        _attach_preview(record, data, filename)
        session.documents.append(record)
        _clear_cached_dossier(session)
        _save_session(session)
        return record

    from company_profile import extract_company_fields

    mapped = [f.to_dict() for f in extract_company_fields(result.text)]
    matched = {entry["field"] for entry in mapped if entry.get("value")}
    # The full entries for what we actually wrote, so the supplier's review
    # list can show label/value/source the way the buyer's auto-fill does.
    # These are the supplier's own declarations coming back to them.
    filled: list[Dict[str, Any]] = []

    # Reconciliation, applied uniformly to every field the mapper recognised
    # rather than only to the fields this document type is "expected" to
    # carry. A licence that names the company should fill an empty legal_name,
    # and a licence that names a *different* company should ask — and neither
    # behaviour should depend on which code the file was filed under.
    #
    # The rule for every field is the same three cases:
    #   blank     -> fill it (never overwrite)
    #   agrees    -> do nothing
    #   disagrees -> ask, and leave the supplier's own value in place
    # Silently rewriting a supplier's declaration would be wrong, and for a bank
    # name or a licence number commercially dangerous.
    for entry in mapped:
        name = entry.get("field")
        from_document = entry.get("value")
        if not name or from_document is None:
            continue
        # Never write a field the supplier does not own, and never one the
        # portal will not show them back.
        if name not in SUPPLIER_VISIBLE_FORM_FIELDS:
            continue

        typed = session.form.get(name)
        if typed is not None and str(typed).strip():
            if not _values_agree(typed, from_document):
                session.clarifications.append(Clarification(
                    field=name,
                    question=_clarification_question(name, from_document),
                ))
            continue

        session.form[name] = from_document
        matched.add(name)
        filled.append({
            "field": name,
            "value": from_document,
            "label": entry.get("label", name),
            "confidence": entry.get("confidence", "exact"),
            "source": entry.get("source", ""),
            "step": entry.get("step", 0),
        })

    status = "Read" if matched else "Read, nothing matched"
    record = {
        "reference": reference,
        "document_code": document_code,
        "filename": filename,
        "status": status,
        "extracted_fields": sorted(matched),
    }
    _attach_preview(record, data, filename)
    session.documents.append(record)
    _clear_cached_dossier(session)
    _save_session(session)
    # Agent Activity: the sub-agent's own record of the read — what arrived,
    # what came out of it, under which form rule. Wording follows the
    # onboarding agent's orchestrator ("Received item … ; no extraction
    # needed") so both consoles read like one system.
    item = str(document_code).strip() or "—"
    summary = (
        f"Read item {item} ({filename}); {len(matched)} field(s) extracted."
        if status == "Read"
        else f"Received item {item} ({filename}). Recorded; no extraction needed."
    )
    log_agent_activity(
        "Document Intelligence",
        summary,
        rule=(f"Form {document_code}" if str(document_code).strip() else ""),
        vendor=session.vendor_name,
        level="L3",
    )
    # `filled` is returned but deliberately not persisted: session.documents
    # holds document evidence, and duplicating every extracted value into it
    # would store the same company data twice and give the read paths a second
    # place to forget to redact.
    return {**record, "filled": filled}


def _values_agree(typed: Any, from_document: Any) -> bool:
    """Compare two values the way the form means them.

    Punctuation and case are ignored, so a licence number typed as
    "CN-1094821" and printed as "cn 1094821" is the same number and must not
    raise a clarification against the supplier. That normalisation also makes
    an IBAN printed with spaces equal to the same IBAN written without, so no
    separate IBAN rule is needed.

    The digit comparison exists for licence and account numbers, where one side
    may carry a prefix the other drops ("CN-1094821" vs "1094821").
    """
    def normalise(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(value)).lower()

    typed_normalised = normalise(typed)
    doc_normalised = normalise(from_document)
    if typed_normalised == doc_normalised:
        return True

    typed_digits = re.sub(r"\D", "", str(typed))
    doc_digits = re.sub(r"\D", "", str(from_document))
    # Both sides must carry digits, and one must be the other's tail: a prefix
    # difference is tolerable, anything else is a real disagreement.
    return bool(
        typed_digits
        and doc_digits
        and (
            typed_digits.endswith(doc_digits)
            or doc_digits.endswith(typed_digits)
        )
    )


# --------------------------------------------------------------------------
# Approvals — GET briefs, POST decides
# --------------------------------------------------------------------------

_APPROVAL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,}$")


def create_approval_brief(
    vendor_name: str,
    summary: str,
    requested_by: str,
    supplier_token: str = "",
) -> str:
    """Create a one-time approval brief and return its token.

    The brief is rendered for a human to read. Deciding happens only through
    `record_approval_decision`, which requires a POST.
    """
    token = secrets.token_urlsafe(24)
    with _APPROVALS_LOCK:
        _APPROVALS[token] = {
            "vendor_name": vendor_name,
            "summary": summary,
            "requested_by": requested_by,
            "supplier_token": supplier_token,
            "decided": False,
            "decision": None,
            "decided_by": None,
            "decided_at": None,
            "created_at": time.time(),
        }
        brief = _APPROVALS[token]
    _persist_item("approval", token, brief)
    return token


def read_approval_brief(token: str) -> Dict[str, Any]:
    """Render a brief. Read-only, and deliberately so.

    A GET is what a corporate mail security gateway, a link previewer or a chat
    client does the moment it sees a URL. If GET could approve, onboarding would
    be approved by the recipient's inbox before anyone read it. Every decision
    therefore requires POST.
    """
    with _APPROVALS_LOCK:
        brief = _APPROVALS.get(token)
    if brief is None:
        raise HTTPException(status_code=404, detail="This approval link is not valid.")
    return {
        "vendor_name": brief["vendor_name"],
        "summary": brief["summary"],
        "requested_by": brief["requested_by"],
        "decided": brief["decided"],
        "decision": brief["decision"],
        "decided_by": brief["decided_by"],
        "decided_at": brief["decided_at"],
        "notice": (
            "Viewing this brief does not approve anything. Approval is "
            "recorded only by an explicit POST from an authorised approver."
        ),
    }


def record_approval_decision(
    token: str, decision: str, approver: str
) -> Dict[str, Any]:
    """Record an approval decision. POST only, authenticated, one-time."""
    decision = (decision or "").strip().lower()
    if decision not in ("approved", "rejected"):
        raise HTTPException(
            status_code=400, detail="Decision must be 'approved' or 'rejected'."
        )
    if not (approver or "").strip():
        raise HTTPException(status_code=401, detail="An authenticated approver is required.")

    with _APPROVALS_LOCK:
        brief = _APPROVALS.get(token)
        if brief is None:
            raise HTTPException(status_code=404, detail="This approval link is not valid.")
        if brief["decided"]:
            raise HTTPException(
                status_code=409, detail="This approval has already been recorded."
            )
        brief["decided"] = True
        brief["decision"] = decision
        brief["decided_by"] = approver
        brief["decided_at"] = time.time()
        decided = dict(brief)

        if brief.get("supplier_token"):
            session = _SESSIONS.get(brief["supplier_token"])
            if session is not None:
                session.neutral_status = (
                    "Approved" if decision == "approved" else "Action required"
                )
                _save_session(session)

    _persist_item("approval", token, decided)
    return read_approval_brief(token)


# --------------------------------------------------------------------------
# Compliance alerts — MLRO only
# --------------------------------------------------------------------------

def create_compliance_alert(
    alert_id: str, vendor_name: str, reason: str, supplier_token: str = ""
) -> None:
    """Raise an alert for MLRO decision. Reason stays internal."""
    with _ALERTS_LOCK:
        _ALERTS[alert_id] = {
            "id": alert_id,
            "vendor_name": vendor_name,
            "reason": reason,
            "supplier_token": supplier_token,
            "decided": False,
            "decision": None,
            "justification": None,
            "decided_by": None,
            "decided_at": None,
            "created_at": time.time(),
        }
        alert = _ALERTS[alert_id]
    _persist_item("alert", alert_id, alert)


def _raise_screening_alert(
    session: Session, tier: str, score: float, signals_bag: Dict[str, Any]
) -> None:
    """Raise an MLRO alert when the score says the MLRO must look.

    Two triggers, matching what the compliance queue exists for:
      - the weighted score landed in the High Risk (EDD) band, or
      - sanctions screening on the submitted form shows a confirmed or
        unresolved (watchlist) match, whatever the tier.

    The reason stays internal — it carries the tier and the score, which are
    exactly what `supplier_view` must never emit. One open alert per token:
    a form re-scored after an edit must not stack a second row in the queue.
    """
    sanctions = signals_bag.get("sanctions") or {}
    hits = []
    if sanctions.get("confirmed"):
        hits.append("confirmed sanctions match")
    if sanctions.get("unresolved_match"):
        hits.append("unresolved sanctions/watchlist match")
    high_tier = "High Risk" in (tier or "")
    if not high_tier and not hits:
        return

    with _ALERTS_LOCK:
        if any(
            alert.get("supplier_token") == session.token and not alert.get("decided")
            for alert in _ALERTS.values()
        ):
            return

    reason = f"Auto-raised at submission: {tier} (weighted score {score:.2f})"
    if hits:
        reason += "; " + ", ".join(hits)
    create_compliance_alert(
        f"auto-{session.token[:16]}",
        session.vendor_name or "A vendor",
        reason,
        session.token,
    )


def _decider_role(claims: Dict[str, Any]) -> str:
    """The caller's normalized role when it is one that may decide, else "".

    Shared by the gate below and the Glass Box entry a decision writes, so the
    role recorded in the audit log is the same string the gate matched.
    """
    role = str(claims.get("role") or claims.get("roles") or "").strip().lower()
    if isinstance(claims.get("roles"), list):
        role = next(
            (str(r).strip().lower() for r in claims["roles"] if str(r).strip().lower() in DECIDER_ROLES),
            "",
        )
    return role if role in DECIDER_ROLES else ""


def _require_mlro(claims: Dict[str, Any]) -> str:
    """Authorise a compliance decision, or refuse it.

    The role comes from the verified ID token's claims, never from a header the
    caller can set. A `X-Role: mlro` header would be a suggestion, and this
    decision releases an escalation to a human being.
    """
    if not claims:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    if not _decider_role(claims):
        raise HTTPException(
            status_code=403,
            detail=(
                "This decision is restricted to the MLRO or the Subsidiary "
                "Compliance Officer. Your role may not record compliance "
                "decisions."
            ),
        )
    return str(claims.get("uid") or claims.get("email") or "mlro")


def decide_compliance_alert(
    alert_id: str,
    decision: str,
    claims: Dict[str, Any],
    justification: str = "",
    assisted: bool = False,
    rag: bool = False,
) -> Dict[str, Any]:
    """Record an MLRO's decision on a compliance alert.

    `justification` is the decider's reasoning. It is persisted on the alert
    (and returned to the console's queue) as the audit trail; the supplier
    never sees it — they are told only that action is required, per the
    neutral-status rule below.

    `assisted` says the justification began as the Compliance Agent's draft
    (the co-pilot's textarea button). It only ever colours the Glass Box
    entry — "Cleared by MLRO (Assisted by Compliance Agent)" versus "Cleared
    by MLRO" — so the log states plainly when a human adopted agent text,
    and never claims assistance that did not happen.

    `rag` says the adopted text came from the co-pilot's RAG pipeline
    (screening findings + document vault + retrieved policy rules). Together
    with `assisted` the marker becomes "Assisted by Compliance Agent (RAG
    Context Applied)"; without the flag the plain assistance marker is kept,
    so the log never claims a retrieval pass that did not happen.
    """
    approver = _require_mlro(claims)

    decision = (decision or "").strip().lower()
    if decision not in ("escalated", "cleared", "approved", "rejected"):
        raise HTTPException(
            status_code=400,
            detail="Decision must be one of escalated, cleared, approved, rejected.",
        )

    # Recorded text is normalized on the way in — entities unescaped,
    # markdown unwrapped, section headings on their own lines — so the
    # queue, the drawer and the textarea all show the same clean document
    # the officer actually filed.
    recorded = _plain_document((justification or "").strip())

    with _ALERTS_LOCK:
        alert = _ALERTS.get(alert_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="No such compliance alert.")
        if alert["decided"]:
            raise HTTPException(
                status_code=409, detail="This alert has already been decided."
            )
        alert["decided"] = True
        alert["decision"] = decision
        alert["justification"] = recorded
        alert["decided_by"] = approver
        alert["decided_at"] = time.time()
        decided_alert = dict(alert)
        token = alert.get("supplier_token")

    if token:
        # Resolve through disk as well as cache: the alert and the session may
        # have been written before a restart that evicted the session from
        # memory, and silently skipping the status change would leave the
        # supplier staring at a stale "In progress".
        session = _find_session(token)
        if session is not None:
            # The supplier learns that something needs attention, and nothing
            # about why. The MLRO's reasoning is not theirs to see.
            session.neutral_status = (
                "Action required" if decision in ("escalated", "rejected") else "Under review"
            )
            _save_session(session)

    _persist_item("alert", alert_id, decided_alert)

    # One Glass Box row per decision, written after the decision is durable:
    # the log describes what happened, so it must not outlive a decision that
    # failed to persist. The role is the one the gate matched, uppercase for
    # the record — MLRO or SCO, never the raw claim.
    role_label = _decider_role(claims).upper()
    verb = {"cleared": "Cleared", "rejected": "Blocked",
            "approved": "Approved", "escalated": "Escalated"}[decision]
    if rag and assisted:
        suffix = " (Assisted by Compliance Agent (RAG Context Applied))"
    elif assisted:
        suffix = " (Assisted by Compliance Agent)"
    else:
        suffix = ""
    log_glassbox(
        f"{verb} by {role_label}{suffix}",
        actor=approver,
        actor_role=role_label,
        actor_type="human",
        level="L0",
        rules=["Policy: MLRO decision is final"],
        alert_id=alert_id,
        vendor_name=decided_alert.get("vendor_name") or "",
        payload={
            "decision": decision,
            "assisted": bool(assisted),
            "rag": bool(rag and assisted),
            "justification": recorded,
            "justification_chars": len(recorded),
        },
    )

    # Agent Activity: what the system did with that human decision. The Glass
    # Box above records who decided; this records the sub-agent bookkeeping —
    # queue closed, supplier's timeline moved on — which is the part a buyer
    # watching the dashboard should see scrolling.
    log_agent_activity(
        "Screening",
        f"Alert on {decided_alert.get('vendor_name') or 'the vendor'} recorded "
        f"as {decision}; compliance queue updated, supplier timeline moved on.",
        rule="Sanctions, PEP and Adverse-Media Screening; R08",
        vendor=decided_alert.get("vendor_name") or "",
        level="L3",
    )

    return {
        "id": alert_id,
        "decided": True,
        "decision": decision,
        "justification": (justification or "").strip(),
        "decided_by": approver,
        "decided_at": alert["decided_at"],
    }


def read_alert(alert_id: str) -> Optional[Dict[str, Any]]:
    with _ALERTS_LOCK:
        alert = _ALERTS.get(alert_id)
    return dict(alert) if alert else None


# --------------------------------------------------------------------------
# Glass Box — the append-only audit log behind the console
#
# Same contract as the onboarding agent's Glass Box: every human and agent
# action, with the level it was taken at and the rule it relied on. L0 is a
# human decision (an agent may not take it), L3 autonomous.
#
# Append-only by construction: rows are inserted under a fresh sequence
# number and no code path updates or rewrites one. The only clear is
# `reset_all()`, which drops the whole demo world — an operator action, not
# an edit of history. In memory-only mode (the test suite) the list is the
# store; with a database the rows ride along in `portal_items`.
# --------------------------------------------------------------------------

_GLASSBOX: List[Dict[str, Any]] = []
_GLASSBOX_LOCK = threading.Lock()


def _merge_restored_rows(
    rows: List[Dict[str, Any]],
    store: List[Dict[str, Any]],
    lock: threading.Lock,
) -> None:
    """Merge persisted seq-keyed rows into an in-memory append-only store.

    `_restore()` is not startup-only: `list_invitations()` re-runs it on every
    control-tower refresh to pick up rows another instance wrote. Approvals and
    alerts survive that because assigning a dict key twice is a no-op — an
    appended row would not be, so merge by the sequence number: rows already
    in memory are skipped, and a row appended but not yet persisted is never
    dropped. The sequence number is each record's own claim to its place.
    """
    with lock:
        known_seqs = {event.get("seq") for event in store}
        for payload in rows:
            if payload.get("seq") not in known_seqs:
                store.append(payload)
        store.sort(key=lambda event: event.get("seq") or 0)


def log_glassbox(
    action: str,
    *,
    actor: str,
    actor_role: str = "",
    actor_type: str = "agent",
    level: str = "L3",
    rules: Optional[List[str]] = None,
    alert_id: str = "",
    vendor_name: str = "",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append one event and return it. Nothing here is ever edited later."""
    # Touch the store first: on a cold process this restores the persisted
    # log, so the sequence number continues the existing log instead of
    # restarting at 1 and colliding with (or overwriting) a stored row.
    _db()
    with _GLASSBOX_LOCK:
        seq = (_GLASSBOX[-1]["seq"] if _GLASSBOX else 0) + 1
        event = {
            "seq": seq,
            "at": time.time(),
            "actor": actor,
            "actor_role": actor_role,
            "actor_type": actor_type,
            "action": action,
            "level": level,
            "rules": rules or [],
            "alert_id": alert_id,
            "vendor_name": vendor_name,
            "payload": payload or {},
        }
        _GLASSBOX.append(event)
    _persist_item("glassbox", str(seq), event)
    return event


def glassbox_events() -> List[Dict[str, Any]]:
    """Every event, oldest first — the order the dossier prints them in."""
    # Touch the store first: on a cold process the persisted log has not been
    # restored yet, and returning the empty in-memory list as "the whole log"
    # would show a blank Glass Box right after a restart.
    _db()
    with _GLASSBOX_LOCK:
        return [dict(event) for event in _GLASSBOX]


# --------------------------------------------------------------------------
# Agent Activity — the background sub-agent stream behind the dashboard
#
# Same append-only contract as the Glass Box, but the cast differs: the Glass
# Box records human decisions (L0), this stream records what the sub-agents
# did on their own — Document Intelligence reading an upload, Supplier
# Concierge chasing a silent supplier, Screening closing a queue. The agent
# names, levels and rule references follow the onboarding agent's
# orchestrator so both consoles read like one system.
#
# An empty store is seeded from `_ACTIVITY_SEED` on first read or first
# write: a presenter opening the dashboard must find history to scroll, and
# `reset_all()` must bring it back. Real actions append after the fixture,
# never over it, so the feed is fixture-then-truth.
# --------------------------------------------------------------------------

_ACTIVITY: List[Dict[str, Any]] = []
_ACTIVITY_LOCK = threading.Lock()

# (seconds ago, sub-agent, level, vendor, summary, rule). Ages rather than
# absolute times, so the fixture always reads chronologically whichever
# moment it is seeded at.
_ACTIVITY_SEED: List[tuple] = [
    (5 * 3600, "Intake and Triage", "L3", "Aurora Medical Supplies FZE",
     "Invitation emailed to Aurora Medical Supplies FZE with a secure link.",
     "Spec 3.4 day 0"),
    (3 * 3600 + 20 * 60, "Document Intelligence", "L3", "Aurora Medical Supplies FZE",
     "Read item 73 (A_73_uae_bank_confirmation.pdf); 4 field(s) extracted.",
     "Form 73"),
    (2 * 3600 + 50 * 60, "Document Intelligence", "L3", "Caspian Energy Trading Ltd",
     "Verified item 68 (C_68_bank_letter.pdf) against the account declared on "
     "Form 73. No discrepancy.",
     "Form 68"),
    (70 * 60, "Document Intelligence", "L3", "Bharat Heavy Fabricators LLC",
     "Received item 74 (B_74_address_change_evidence.pdf). Recorded; no "
     "extraction needed.",
     "Form 74"),
    (40 * 60, "Supplier Concierge", "L3", "Aurora Medical Supplies FZE",
     "Email clarification chase sent to Aurora Medical Supplies FZE (day 3): "
     "onboarding form still outstanding.",
     "Spec 3.4; A7"),
    (18 * 60, "Screening", "L0", "Caspian Energy Trading Ltd",
     "Screened 4 subjects against 11 lists (internal). Alerts: 1.",
     "Sanctions, PEP and Adverse-Media Screening; R08"),
]


def _seed_activity_locked() -> List[Dict[str, Any]]:
    """Fill an empty store from the demo fixture. Caller holds the lock."""
    if _ACTIVITY:
        return []
    now = time.time()
    seeded = [
        {
            "seq": seq,
            "at": now - age,
            "agent": agent,
            "level": level,
            "summary": summary,
            "rule": rule,
            "vendor": vendor,
        }
        for seq, (age, agent, level, vendor, summary, rule) in enumerate(
            _ACTIVITY_SEED, start=1
        )
    ]
    _ACTIVITY.extend(seeded)
    return seeded


def log_agent_activity(
    agent: str,
    summary: str,
    rule: str = "",
    *,
    vendor: str = "",
    level: str = "L3",
) -> Dict[str, Any]:
    """Append one sub-agent event and return it.

    Append-only, like the Glass Box: no code path edits a row. The fixture
    seeds first when the store is empty, so a live action never claims seq 1
    in a demo world that should have started with history.
    """
    # Touch the store first: a cold process must continue the persisted
    # stream, not restart its sequence at 1 over the stored rows.
    _db()
    with _ACTIVITY_LOCK:
        pending = _seed_activity_locked()
        seq = (_ACTIVITY[-1]["seq"] if _ACTIVITY else 0) + 1
        event = {
            "seq": seq,
            "at": time.time(),
            "agent": agent,
            "level": level,
            "summary": summary,
            "rule": rule,
            "vendor": vendor,
        }
        _ACTIVITY.append(event)
        pending.append(event)
    for row in pending:
        _persist_item("agent_activity", str(row["seq"]), row)
    return event


def agent_activity_events() -> List[Dict[str, Any]]:
    """Every event, oldest first — the panel reverses it for newest-first."""
    # Touch the store first: on a cold process the persisted stream has not
    # been restored yet, and seeding here would then append a second copy of
    # the fixture on top of it.
    _db()
    with _ACTIVITY_LOCK:
        pending = _seed_activity_locked()
        events = [dict(event) for event in _ACTIVITY]
    for row in pending:
        _persist_item("agent_activity", str(row["seq"]), row)
    return events


# --------------------------------------------------------------------------
# Agentic Decision Co-Pilot — the Compliance Agent's draft analysis
# --------------------------------------------------------------------------

def _screening_findings(alert: Dict[str, Any]) -> List[str]:
    """The evidence behind an alert, extracted from the submission record.

    Deterministic on purpose: everything quoted here already exists on disk —
    the queue's reason, the score the engine produced, the sanctions flags
    the supplier declared, the documents extraction actually read. The agent
    drafts *from* the record; it never invents one, which is what makes the
    draft safe to show as a proposal a human can verify line by line.
    """
    findings: List[str] = [
        f"Queue reason: {alert.get('reason') or 'raised by screening'}"
    ]
    token = alert.get("supplier_token")
    session = _find_session(token) if token else None
    if session is None:
        return findings

    internal = session.internal or {}
    score = internal.get("weighted_risk_score")
    tier = internal.get("assigned_risk_tier")
    if score is not None and tier:
        due = internal.get("due_diligence_level") or ""
        findings.append(
            f"Weighted risk score {float(score):.2f} — {tier}"
            + (f" ({due})" if due else "")
        )

    import signals as signals_mod

    sanctions = signals_mod.build_signals(dict(session.form)).get("sanctions") or {}
    if sanctions.get("confirmed"):
        findings.append(
            "Declared sanctions screening: a confirmed match is declared on the form."
        )
    elif sanctions.get("unresolved_match"):
        findings.append(
            "Declared sanctions screening: an unresolved (watchlist) match "
            "is declared on the form."
        )
    else:
        findings.append(
            "Declared sanctions screening: no confirmed match declared on the form."
        )

    # Entity-name correspondence: the false-positive argument stands or falls
    # on whether the alert's name and the filed legal name are the same legal
    # person — quoted verbatim so the draft can reference the exact strings.
    form = dict(session.form)
    legal = str(form.get("legal_name") or "").strip()
    named = str(alert.get("vendor_name") or "").strip()
    if legal and named:
        if legal.casefold() == named.casefold():
            findings.append(
                f"Entity name: the alert and the form's legal name agree "
                f"('{legal}')."
            )
        else:
            findings.append(
                f"Entity name variation: the alert names '{named}'; the form "
                f"files '{legal}' as the legal name."
            )

    # PEP status is whatever the supplier declared — every PEP-ish field on
    # the form, quoted as filed; nothing is inferred from an absent key.
    pep = [
        f"{key} = {value}"
        for key, value in sorted(form.items())
        if "pep" in str(key).lower() and value not in (None, "", [], {})
    ]
    findings.append(
        "PEP status: "
        + ("; ".join(pep) if pep else "no PEP declaration on the form.")
    )

    factors = internal.get("factors") or []

    def _weighted(factor: Dict[str, Any]) -> Decimal:
        try:
            return Decimal(str(factor.get("weighted") or "0"))
        except Exception:
            return Decimal("0")

    top = sorted(factors, key=_weighted, reverse=True)[:3]
    if top:
        findings.append("Highest-scoring factors:")
        for index, factor in enumerate(top, start=1):
            findings.append(
                f"  {index}. {factor.get('label')} — rule: {factor.get('rule')}"
            )

    read_docs = [
        d
        for d in session.documents or []
        if str(d.get("status") or "").startswith("Read")
    ]
    for doc in read_docs[:4]:
        findings.append(
            f"Document evidence: Doc #{doc.get('document_code')} — "
            f"{doc.get('reference')} ({doc.get('status')})"
        )
    return findings


def _compose_draft(
    alert: Dict[str, Any], context: Dict[str, List[str]], decision: str
) -> str:
    """A filing-ready draft in document layout the decider can adopt, edit
    or throw away.

    The same three legs the RAG context is built from, laid out as a
    document rather than one flat bullet list: a headline finding, numbered
    sections (screening findings & watchlist alerts, extracted document
    vault, policy specification rules), the recommendation, and the advisory
    trailer. Advisory by construction — it opens with "AI Finding:", offers
    a recommendation rather than a verdict, and closes by naming the human
    who must still record the decision. The button that requests it only
    ever fills a textarea; filing stays behind the 20-character gate and
    the decider's own click.
    """
    vendor = alert.get("vendor_name") or "the vendor"
    if decision == "cleared":
        headline = (
            f"AI Finding: the screening alert against {vendor} presents as "
            "a false positive on the record as filed."
        )
        recommendation = (
            "Recommendation: clear the match as a false positive — no name "
            "or watchlist correspondence is corroborated by the submission "
            "record — subject to your review."
        )
    else:
        headline = (
            f"AI Finding: the screening alert against {vendor} supports "
            "blocking the vendor."
        )
        recommendation = (
            "Recommendation: block the vendor pending enhanced due diligence "
            "— the declared match and risk tier meet the policy bar for "
            "refusal — subject to your review."
        )

    sections = [
        (
            "1. Screening & Watchlist Alert Summary",
            context.get("screening") or [],
        ),
        (
            "2. Document Evidence Cross-Verification",
            context.get("documents") or [],
        ),
        ("3. Policy & Compliance Basis", context.get("policy") or []),
    ]
    blocks = []
    for heading, lines in sections:
        if not lines:
            continue
        # Lines may carry their own "- " or "  N." prefix (the policy leg
        # writes sub-bullets); keep the content, drop the top-level dash so
        # the section never renders "- - …".
        body = "\n".join(
            str(line)[2:] if str(line).startswith("- ") else str(line)
            for line in lines
        )
        blocks.append(f"{heading}\n{body}")
    blocks.append(f"4. AI Recommendation\n{recommendation}")
    return "\n\n".join(
        [
            headline,
            *blocks,
            "Compliance Agent draft — advisory only. The decision and its "
            "justification remain with the reviewing compliance officer "
            "(human-in-the-loop).",
        ]
    )


def _document_vault(session: Any) -> List[str]:
    """The second RAG leg: what extraction read out of the supplier's vault.

    Entity, tax, bank and ownership facts beside the documents they came
    from — quoted from the form as filed, "not on file" only where the
    field really is empty. Everything here is what the draft may reference;
    none of it is inferred.
    """
    form = dict(session.form)

    def _f(*keys: str) -> str:
        for key in keys:
            value = form.get(key)
            if value not in (None, "", [], {}):
                return str(value).strip()
        return "not on file"

    lines = [
        f"Trade licence: {_f('trade_license_no')} "
        f"(expiry {_f('trade_license_expiry')}).",
        f"VAT / TRN: {_f('vat_registration_no')} "
        f"(status: {_f('vat_registration_status')}).",
        f"Bank account: {_f('bank_account_name')} / "
        f"{_f('bank_account_number', 'bank_iban')} "
        f"(bank {_f('bank_name_branch_country')}).",
        f"Registered address: {_f('registered_address')}.",
        f"Authorized representative: {_f('authorized_representative_name')} "
        f"(ID {_f('authorized_representative_id', 'authorized_representative_id_expiry')}).",
    ]
    ubos = form.get("ubos")
    if isinstance(ubos, list):
        ubo_text = f"{len(ubos)} beneficial owner(s) declared"
        if ubos:
            ubo_text += ": " + "; ".join(str(u) for u in ubos[:4])
    elif ubos not in (None, "", [], {}):
        ubo_text = str(ubos)
    else:
        ubo_text = "none declared"
    lines.append(f"UBO declarations: {ubo_text}.")
    declarations = [
        f"{key} = {value}"
        for key, value in sorted(form.items())
        if any(tag in str(key).lower() for tag in ("nominee", "bearer"))
        and value not in (None, "", [], {})
    ]
    if declarations:
        lines.append("Ownership declarations: " + "; ".join(declarations) + ".")

    docs = list(session.documents or [])
    for doc in docs[:8]:
        fields = ", ".join(str(x) for x in (doc.get("extracted_fields") or []))
        lines.append(
            f"Doc #{doc.get('document_code')} {doc.get('reference')} — "
            f"{doc.get('status')}" + (f"; extracted: {fields}" if fields else "")
        )
    if not docs:
        lines.append("No documents uploaded — nothing has been extracted yet.")
    return lines


def _policy_rules(session: Any, decision: str) -> List[str]:
    """The third RAG leg: the specification rules, quoted locally.

    The corpus is the authority — the generation pass retrieves the policy
    text itself — but the Risk Scoring Matrix bands, the Always-EDD floor
    and the trigger codes are deterministic, so the draft can cite them by
    name even when corpus retrieval is unreachable.
    """
    import risk_engine
    import triggers as triggers_mod

    lines = [
        f"Policy: {risk_engine.POLICY_SECTION_MATRIX} (specification):",
    ]
    for upper, label, dd_level, cycle in risk_engine.TIER_BANDS:
        lines.append(
            f"- weighted score ≤ {upper} → {label}: {dd_level}; refresh {cycle}."
        )
    lines.append(
        f"- Always-EDD floor: a mandatory trigger raises the score to at "
        f"least {triggers_mod.FORCED_EDD_FLOOR} ({risk_engine.TIER_HIGH})."
    )
    codes = []
    for trigger in triggers_mod.MANDATORY_EDD_TRIGGERS or ():
        code = getattr(trigger, "code", None)
        label = getattr(trigger, "label", None)
        codes.append(
            f"{code} ({label})" if code and label else str(code or trigger)
        )
    if codes:
        lines.append("- Always-EDD triggers:")
        for index, entry in enumerate(codes, start=1):
            lines.append(f"  {index}. {entry}")
    internal = session.internal or {}
    score = internal.get("weighted_risk_score")
    tier = internal.get("assigned_risk_tier")
    if score is not None and tier:
        lines.append(f"- This submission scored {float(score):.2f} → {tier}.")
    if decision == "cleared":
        lines.append(
            "- Clearing a screening match requires a documented false "
            "positive: the record must show why the correspondence fails "
            "(different legal person, verified registration evidence, an "
            "exact document reference)."
        )
    else:
        lines.append(
            "- Blocking cites the screening hit and the control it breaches; "
            "unverified documentation alone supports a hold, never a pass."
        )
    return lines


_DRAFT_SYSTEM = (
    "You are the Compliance Agent at National Holding, drafting justification "
    "text for a Money Laundering Reporting Officer (MLRO) before they record "
    "a decision on a screening alert. A Vertex AI RAG corpus holds the "
    "National Holding Procurement Policy: retrieve the passages that govern "
    "this decision (Form 73/74 requirements, Spec 3.4, Assumption A7, the "
    "SDD/CDD/EDD thresholds of the Risk Scoring Matrix) and cite section "
    "names exactly as retrieved — never strengthen a rule beyond the policy "
    "text or the record below.\n"
    'Output rules: begin with the exact words "AI Finding:" followed by the '
    "headline finding as one short paragraph. Then write the justification "
    "in document layout — plain-text numbered section headings "
    '"1. Screening & Watchlist Alert Summary" (entity match, score details), '
    '"2. Document Evidence Cross-Verification" (trade licence, address '
    "proof, bank confirmation), "
    '"3. Policy & Compliance Basis" (KYC/EDD rules, Form 73/74 alignment), '
    "each heading on its own line with one blank line before and after it, "
    "each section one short paragraph (no bullet lists), citing the record: "
    "exact document dates and numbers, entity-name correspondence or "
    "mismatch, verified registration evidence, named screening hits, and "
    "the policy control each one breaches. Leave out any section that adds "
    "nothing, then always close with a final section "
    '"4. AI Recommendation" whose paragraph carries the '
    '"Recommendation: … subject to your review." conclusion, followed by '
    'the line "Compliance Agent draft — advisory only. The decision and '
    "its justification remain with the reviewing compliance officer "
    '(human-in-the-loop)." Plain text only — never markdown: no ** or __ '
    'around headings, no "#", no backticks.'
)


def _draft_prompt(
    vendor: str, decision: str, context: Dict[str, List[str]]
) -> str:
    """The record under review plus the ask, laid out for the model."""
    if decision == "cleared":
        ask = (
            "Draft the false-positive justification: explain WHY this hit is "
            "not the vendor — exact document dates and numbers, the "
            "entity-name correspondence or mismatch, and verified "
            "registration evidence from the vault."
        )
    else:
        ask = (
            "Draft the blocking justification: name the specific screening "
            "hit, any unverified documentation, and the policy rule it "
            "violates (section or trigger code)."
        )
    parts = [f"Vendor under review: {vendor}", ask, ""]
    for title, key in (
        ("Screening findings and watchlist alerts", "screening"),
        ("Extracted document vault", "documents"),
        ("Policy specification rules", "policy"),
    ):
        parts.append(f"## {title}")
        parts.extend(f"- {line}" for line in context.get(key) or [])
        parts.append("")
    parts.append(
        "Retrieve the procurement policy passages for Form 73/74 "
        "requirements, Spec 3.4, Assumption A7 and the EDD thresholds from "
        "the corpus, and cite them by section name."
    )
    return "\n".join(parts)


def _llm_rag_draft(
    vendor: str, decision: str, context: Dict[str, List[str]]
) -> str:
    """One grounded generation pass over the three-part retrieval context.

    The policy corpus is bound as a retrieval tool, so the model pulls the
    governing passages (Form 73/74, Spec 3.4, Assumption A7, EDD thresholds)
    itself; the screening findings and the document vault ride along in the
    prompt as the record under review. Runs on a daemon thread with a hard
    budget (`RAG_DRAFT_TIMEOUT`, default 15s) so a slow or hung call can
    never pin the request — and raises on anything that goes wrong (no
    credential, no network, empty generation) so the caller falls back to
    the deterministic record draft.
    """
    import os
    import threading

    budget = float(os.environ.get("RAG_DRAFT_TIMEOUT", "45"))
    outcome: Dict[str, Any] = {}

    def _generate() -> None:
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, Tool
            from vertexai.preview import rag

            vertexai.init(
                project=os.environ.get("PROJECT_ID", "test-rag-corpus-project"),
                location=os.environ.get("LOCATION", "us-west1"),
            )
            corpus = os.environ.get(
                "RAG_CORPUS",
                "projects/test-rag-corpus-project/locations/us-west1/"
                "ragCorpora/137359788634800128",
            )
            tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[rag.RagResource(rag_corpus=corpus)],
                        similarity_top_k=8,
                    )
                )
            )
            model = GenerativeModel(
                os.environ.get("RAG_DRAFT_MODEL", "gemini-2.5-flash"),
                system_instruction=_DRAFT_SYSTEM,
                tools=[tool],
            )
            outcome["text"] = model.generate_content(
                _draft_prompt(vendor, decision, context)
            ).text
        except BaseException as exc:  # noqa: BLE001 — any failure falls back
            outcome["error"] = exc

    worker = threading.Thread(target=_generate, daemon=True)
    worker.start()
    worker.join(budget)
    if worker.is_alive():
        raise TimeoutError(f"RAG generation took longer than {budget:g}s.")
    error = outcome.get("error")
    if error is not None:
        raise RuntimeError(f"RAG generation failed: {error}")
    text = str(outcome.get("text") or "").strip()
    if not text:
        raise ValueError("RAG generation returned nothing.")
    return text


def _plain_document(text: str) -> str:
    """Normalize model output into the plain document the textarea shows.

    The model sometimes speaks markdown — `**bold**` section headings, `#`
    prefixes, deep paragraph indentation — which a textarea renders
    literally, asterisks and all. Unwrap the markers, flush the model's
    paragraph indent (keeping real sub-items), and give every section
    heading the ChatGPT-style breathing room: a blank line before and
    after, never squeezed against its paragraph.
    """
    import html
    import re

    out = html.unescape(str(text or ""))
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)
    out = re.sub(r"__(.+?)__", r"\1", out)
    out = re.sub(r"`([^`\n]+)`", r"\1", out)
    out = re.sub(r"(?m)^#{1,6}[ \t]*", "", out)
    out = out.replace("**", "")  # any stray unmatched marker
    # Single-line model output glues section headings into the prose —
    # "…watchlist entry. 1. Screening & Watchlist Alert Summary The vendor…"
    # — so give each heading, the recommendation and the trailer a line of
    # their own. Idempotent: already-separated text collapses to itself, and
    # the heading pass below restores any blank line this consumes.
    out = re.sub(r"\s+(?=[1-4]\.\s+[A-Z&])", "\n", out)
    out = re.sub(r"\s+(?=(?:Recommendation:|Compliance Agent draft))", "\n", out)
    # …and cut after the section title itself when the body rode along on
    # the heading line: "1. Screening findings & watchlist alerts The vendor…"
    out = re.sub(
        r"(?m)^([1-4]\.\s+(?:"
        r"Screening & Watchlist Alert Summary"
        r"|Screening findings & watchlist alerts"
        r"|Document Evidence Cross-Verification|Document evidence"
        r"|Extracted document vault"
        r"|Policy & Compliance Basis|Policy basis|Policy specification rules"
        r"|AI Recommendation"
        r"))\s+",
        r"\1\n",
        out,
    )

    lines = []
    for raw in out.splitlines():
        line = raw.rstrip()
        body = line.lstrip()
        indent = len(line) - len(body)
        is_item = bool(re.match(r"(?:\d+\.|[-•*])[ \t]", body))
        if indent >= 4 and not is_item:
            line = body  # the model's paragraph indent → flush left
        lines.append(line)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)

    # A heading never touches the paragraph it introduces or the one above.
    heading = (
        r"(?:\d+\.\s+\S[^\n]*|Recommendation:[^\n]*"
        r"|Compliance Agent draft[^\n]*)"
    )
    out = re.sub(r"([^\n])\n(" + heading + ")", r"\1\n\n\2", out)
    out = re.sub(r"(" + heading + r")\n(?=\S)", r"\1\n\n", out)
    return out.strip()


def draft_compliance_analysis(
    alert_id: str, claims: Dict[str, Any], decision: str = "cleared"
) -> Dict[str, Any]:
    """The co-pilot's RAG draft analysis for one alert.

    Decider-only (the same gate as the decision it feeds) and read-only:
    it records nothing and changes nothing. The retrieval context is
    assembled in three legs — screening findings & watchlist alerts, the
    extracted document vault, the policy specification rules — and handed
    to the model with the policy corpus bound as the retrieval tool, so
    the draft quotes the record and the policy rather than the model's
    priors. If generation is unreachable the deterministic record draft
    comes back instead (source "record-fallback"); either way the human
    files or replaces the text, and the decision endpoint is what the
    Glass Box logs.
    """
    _require_mlro(claims)
    decision = (decision or "").strip().lower()
    if decision not in ("cleared", "rejected"):
        raise HTTPException(
            status_code=400,
            detail="A draft can only be written for cleared or rejected.",
        )
    with _ALERTS_LOCK:
        alert = _ALERTS.get(alert_id)
        snapshot = dict(alert) if alert else None
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No such compliance alert.")
    if snapshot.get("decided"):
        raise HTTPException(
            status_code=409, detail="This alert has already been decided."
        )
    findings = _screening_findings(snapshot)
    token = snapshot.get("supplier_token")
    session = _find_session(token) if token else None
    if session is None:
        vault = [
            "No supplier session is linked to this alert — only the queue "
            "record is available."
        ]
        policy: List[str] = []
    else:
        vault = _document_vault(session)
        policy = _policy_rules(session, decision)
    context = {"screening": findings, "documents": vault, "policy": policy}
    vendor = snapshot.get("vendor_name") or "the vendor"
    try:
        draft = _llm_rag_draft(vendor, decision, context)
        source = "rag-llm"
    except Exception:
        draft = _compose_draft(snapshot, context, decision)
        source = "record-fallback"
    # One choke point: whichever leg drafted, the textarea gets clean plain
    # text — markdown never reaches a human who has to edit it.
    draft = _plain_document(draft)
    return {
        "draft": draft,
        "findings": findings,
        "context": context,
        "source": source,
    }


# --------------------------------------------------------------------------
# Submission — score it, tell the buyer, keep it away from the supplier
# --------------------------------------------------------------------------

def run_assessment(session: Session) -> Dict[str, Any]:
    """Score the session with the deterministic core and file it internally.

    Deliberately the scoring chain and not `agent.qualify()`. `qualify` spends
    two LLM passes to produce the narrative dossier a human reads on the review
    screen, and it is the buyer's job to open it. All the notification needs is
    the tier, and the tier comes out of the same `build_signals →
    score_factors → weighted_score → tier_for` path in microseconds, with no
    model call to fail, rate-limit or cost money on a webhook.

    The result lands in `session.internal`, which `supplier_view()` cannot
    reach. Auto-running the score is safe precisely because the supplier is
    never shown it.
    """
    import risk_engine
    import signals as signals_mod

    form = dict(session.form)
    bag = signals_mod.build_signals(form)
    factors = risk_engine.score_factors(bag)
    score = risk_engine.weighted_score(factors)
    tier, due_diligence, cycle = risk_engine.tier_for(score)

    internal = dict(session.internal)
    internal.update({
        "weighted_risk_score": float(score),
        "assigned_risk_tier": tier,
        "due_diligence_level": due_diligence,
        "review_cycle": cycle,
        # FactorResult carries factor/label/weight/score/rule/basis/signal.
        # Attribute names matter here: an earlier version reached for
        # `raw_score`/`weighted`/`rationale`, which do not exist, so every
        # factor rendered with three blank columns and the buyer's "why this
        # tier" table looked broken rather than wrong.
        "factors": [
            {
                "factor": f.factor,
                "label": f.label,
                "score": str(f.score),
                "weight": str(f.weight),
                "weighted": str((Decimal(f.weight) * Decimal(str(f.score))).quantize(Decimal("0.0001"))),
                "rule": f.rule,
                "basis": f.basis,
                "signal": f.signal,
            }
            for f in factors
        ],
        "assessed_at": time.time(),
        "auto_scored": True,
    })
    session.internal = internal

    # Compliance sync: a high-risk tier or an unresolved sanctions hit is the
    # MLRO's to see, not the buyer's dashboard alone. Every submission scores
    # through here, so this is the single point where "high risk or watchlist
    # hit" becomes an alert in the compliance queue; the dedupe keeps one open
    # alert per token however many times the form is re-scored.
    _raise_screening_alert(session, tier, float(score), bag)
    return internal


def submit_session(token: str, notify: bool = True) -> Dict[str, Any]:
    """Mark a session submitted, score it, and notify the inviter.

    Idempotent by design. A supplier double-clicking Submit, or a retried
    request over a flaky link, must not produce two emails announcing the same
    submission, so a session that has already been submitted returns its
    original outcome without rescoring or resending.
    """
    session = _load_session(token)

    if session.submitted_at is not None:
        return {
            "vendor_name": session.vendor_name,
            "submitted_at": session.submitted_at,
            "already_submitted": True,
            "status": session.neutral_status,
            "assessment": _assessment_summary(session.internal),
            "notification": session.internal.get("notification") or {"sent": False,
                                                                   "detail": "not sent"},
        }

    session.submitted_at = time.time()
    # "Under review" is the only honest thing to show: the buyer now has it,
    # and the supplier learns nothing about what it says.
    session.neutral_status = "Under review"

    assessment = run_assessment(session)
    summary = _assessment_summary(assessment)

    # The notification goes to whoever invited the supplier, so it links to the
    # buyer's report of the submission — not to the supplier's form. The email
    # says "Review it here", so the link has to be the thing being reviewed.
    link = report_link(token) if os.environ.get("SUPPLIER_PORTAL_BASE_URL") else ""
    if notify:
        try:
            from services import email as email_service

            result = email_service.notify_submission(
                session.invited_by or None,
                session.vendor_name or "A vendor",
                summary["assigned_risk_tier"],
                summary["weighted_risk_score"],
                summary["due_diligence_level"],
                link,
            )
            notification_recipient = session.invited_by or ""
            notification_detail = getattr(result, "detail", "") or ""
            notification_sent = bool(getattr(result, "sent", False))
        except Exception as exc:  # noqa: BLE001 - never lose a submission to mail
            result = {"sent": False, "detail": f"{type(exc).__name__}: {exc}",
                      "recipient": ""}
            notification_recipient = ""
            notification_detail = f"{type(exc).__name__}: {exc}"
            notification_sent = False
        # Normalise to a dict. The dataclass is not subscriptable, and it goes
        # into the persisted session, so anything left un-normalised is a
        # restart-shaped bug rather than a test-shaped one.
        if hasattr(result, "as_dict"):
            result = result.as_dict()
    else:
        result = {"sent": False, "detail": "notification suppressed", "recipient": ""}
        notification_recipient = ""
        notification_detail = "notification suppressed"
        notification_sent = False

    session.internal = dict(session.internal, notification=result)
    _save_session(session)

    try:
        from main import log_notification
        log_notification(
            "submission",
            f"Vendor submitted their onboarding form: {session.vendor_name or 'A vendor'}",
            notification_recipient,
            notification_sent,
            notification_detail,
            token=session.token,
            link=link,
        )
    except Exception:
        pass

    return {
        "vendor_name": session.vendor_name,
        "submitted_at": session.submitted_at,
        "already_submitted": False,
        "status": session.neutral_status,
        "assessment": summary,
        "notification": result,
    }


def _assessment_summary(internal: Dict[str, Any]) -> Dict[str, Any]:
    """The four fields a buyer is notified about. Nothing else crosses the line."""
    return {
        "weighted_risk_score": f"{float(internal.get('weighted_risk_score', 0)):.2f}",
        "assigned_risk_tier": internal.get("assigned_risk_tier", ""),
        "due_diligence_level": internal.get("due_diligence_level", ""),
    }


# --------------------------------------------------------------------------
# Buyer-side reads
#
# Separate from `supplier_view` on purpose. That one is reachable by anyone
# holding a link and must never contain a conclusion. This one is behind
# Firebase authentication and is the only path that exposes the assessment.
# --------------------------------------------------------------------------

def list_submissions() -> List[Dict[str, Any]]:
    """Every live session, newest first. Buyer-authenticated at the route."""
    now = time.time()
    rows: List[Dict[str, Any]] = []
    with _SESSIONS_LOCK:
        sessions = list(_SESSIONS.values())
    for session in sessions:
        if session.is_expired(now):
            continue
        rows.append({
            "token": session.token,
            "vendor_name": session.vendor_name,
            "status": session.neutral_status,
            "invited_by": session.invited_by,
            "created_at": session.created_at,
            "submitted_at": session.submitted_at,
            "has_submitted": session.submitted_at is not None,
            "document_count": len(session.documents),
            "clarification_count": len(session.clarifications),
            "assigned_risk_tier": session.internal.get("assigned_risk_tier"),
            "weighted_risk_score": session.internal.get("weighted_risk_score"),
            "due_diligence_level": session.internal.get("due_diligence_level"),
        })
    rows.sort(key=lambda r: (r["submitted_at"] or 0, r["created_at"]), reverse=True)
    return rows


def approver_from(decoded: Dict[str, Any]) -> str:
    """The identity to attribute a decision to, taken from verified claims.

    Never from the request body: an audit record that names whoever the caller
    typed is not an audit record.
    """
    return str((decoded or {}).get("email") or (decoded or {}).get("uid") or "")


def record_submission_decision(
    token: str,
    decision: str,
    approver: str,
    justification: str = "",
) -> Dict[str, Any]:
    """An internal reviewer approves or rejects a submitted form.

    This is the decision a buyer records while looking at a submission, which is
    a different thing from `record_approval_decision`. That one answers "has the
    person who was emailed the brief decided yet?", is bound to a one-time token,
    and cannot be revisited once answered. A submission has to survive being
    rejected and resubmitted, and two reviewers disagreeing, so the decision
    lives on the session with its own history instead of being consumed.

    Three deliberate constraints:

    - The decision never touches the score or the tier. Those are the output of
      `risk_engine`; a reviewer releasing them is not permitted to edit them, so
      approving does not make a High Risk submission Low Risk.
    - The supplier learns only the neutral status. A rejection sets "Action
      required", which tells them there is something to fix without saying what,
      what was found, or what it scored.
    - A rejection must say why. The supplier is told to act without being told
      what to act on, and the questions they can see are the ones document
      extraction raised, which are frequently not the reason for rejection.
    """
    normalised = (decision or "").strip().lower()
    if normalised not in ("approved", "rejected"):
        raise HTTPException(
            status_code=400, detail="Decision must be 'approved' or 'rejected'."
        )
    if not (approver or "").strip():
        raise HTTPException(
            status_code=401, detail="An authenticated approver is required."
        )

    reason = (justification or "").strip()
    if normalised == "rejected" and len(reason) < MIN_REJECTION_REASON:
        raise HTTPException(
            status_code=400,
            detail=(
                "A rejection needs a reason of at least "
                f"{MIN_REJECTION_REASON} characters. The supplier is told to act "
                "on it, and this is the only place that says what to act on."
            ),
        )

    session = _load_session(token)
    if session.submitted_at is None:
        raise HTTPException(
            status_code=409,
            detail="This supplier has not submitted, so there is nothing to decide.",
        )

    record = {
        "decision": normalised,
        "justification": reason,
        "decided_by": approver,
        "decided_at": time.time(),
        # Kept from the dossier so the buyer can see what was being approved.
        "vendor_id": session.internal.get("vendor_id", ""),
        "assigned_risk_tier": session.internal.get("assigned_risk_tier"),
        "weighted_risk_score": session.internal.get("weighted_risk_score"),
    }

    history = list(session.internal.get("decision_history") or [])
    history.append(record)
    session.internal = dict(
        session.internal, decision=record, decision_history=history
    )
    session.neutral_status = "Approved" if normalised == "approved" else "Action required"

    notification = _notify_supplier_decision(session, normalised, reason)
    record = dict(record, notification=notification)

    _save_session(session)

    if normalised == "rejected":
        # Agent Activity: the Concierge's neutral-worded return trip — the
        # supplier is told to act without being told what was found, under
        # the same Spec 3.4 / A7 rule the autonomous chaser runs on. An
        # approval is the reviewer's story for the Glass Box, not the
        # sub-agent's for this stream.
        log_agent_activity(
            "Supplier Concierge",
            f"Clarification chase sent to {session.vendor_name or 'the supplier'}: "
            "submission returned for correction; supplier status → Action required.",
            rule="Spec 3.4; A7",
            vendor=session.vendor_name,
            level="L3",
        )
    return record


def _notify_supplier_decision(
    session: Session, decision: str, reason: str
) -> Dict[str, Any]:
    """Tell the supplier the outcome. Never lets a failure lose the decision.

    A rejection that is not delivered is a supplier who is never told to fix
    their form, so this is attempted on every decision rather than left to a
    separate job. The result is recorded on the decision either way: a silent
    send failure must not turn into an unrecorded one.
    """
    if not session.supplier_email:
        return {
            "sent": False,
            "detail": (
                "No supplier email on file. This submission was invited without "
                "one, so the outcome could not be delivered."
            ),
            "recipient": "",
        }
    if not os.environ.get("SUPPLIER_PORTAL_BASE_URL"):
        return {
            "sent": False,
            "detail": "SUPPLIER_PORTAL_BASE_URL is unset, so no link could be built.",
            "recipient": session.supplier_email,
        }

    try:
        from services import email as email_service

        result = email_service.notify_supplier_decision(
            session.supplier_email,
            session.vendor_name or "",
            decision == "approved",
            reason,
            invitation_link(session.token),
        )
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    except Exception as exc:  # noqa: BLE001 - the decision is already recorded
        payload = {
            "sent": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "recipient": session.supplier_email,
        }

    return payload


def submission_decision(token: str) -> Optional[Dict[str, Any]]:
    """The current decision on a submission, or None if undecided."""
    return _load_session(token).internal.get("decision")


def buyer_view(token: str) -> Dict[str, Any]:
    """A submitted form exactly as the supplier sent it, plus the assessment.

    The form is returned verbatim rather than re-shaped: the buyer's question
    when reviewing is "what did this supplier actually declare", and any
    reshaping here is one more place for the answer to drift from the source.
    """
    session = _load_session(token)
    return {
        "token": session.token,
        "vendor_name": session.vendor_name,
        "status": session.neutral_status,
        "statuses_you_may_see": list(NEUTRAL_STATUSES),
        "invited_by": session.invited_by,
        "created_at": session.created_at,
        "submitted_at": session.submitted_at,
        "has_submitted": session.submitted_at is not None,
        # As declared, unsummarised.
        "form": dict(session.form),
        # Metadata only — the raw record may carry a base64 preview copy that
        # only the document route serves, and this view is fetched on every
        # detail, decision and notification build.
        "documents": [
            dict(without_preview(d), filename=document_filename(d))
            for d in session.documents
        ],
        "clarifications": [asdict(c) for c in session.clarifications],
        # Internal. Reachable only through this authenticated function.
        "assessment": {
            "weighted_risk_score": session.internal.get("weighted_risk_score"),
            "assigned_risk_tier": session.internal.get("assigned_risk_tier"),
            "due_diligence_level": session.internal.get("due_diligence_level"),
            "review_cycle": session.internal.get("review_cycle"),
            "factors": session.internal.get("factors", []),
            "assessed_at": session.internal.get("assessed_at"),
            "auto_scored": session.internal.get("auto_scored", False),
        },
        # The full qualification dossier — Appendix F, RAG citations, required
        # controls, narrative — is a separate, slower pipeline from the factor
        # arithmetic above. It is computed on demand rather than at submit, so a
        # supplier is never held waiting on two RAG passes to save a form.
        "report": {
            "ready": bool(session.internal.get("dossier")),
            "computed_at": session.internal.get("dossier_computed_at"),
        },
        # The internal approve/reject, and its history. `status` above is the
        # same fact as seen by the supplier, so a buyer never has to guess
        # whether "Approved" came from a decision or from something else.
        "decision": session.internal.get("decision"),
        "decision_history": list(session.internal.get("decision_history") or []),
        "notification": session.internal.get("notification"),
    }


def qualification_payload(session: Session) -> Dict[str, Any]:
    """The session's form, in the shape the qualification pipeline reads.

    `session.form["documents"]` is not the evidence. Evidence lives in
    `session.documents`, one record per upload, because the wizard needs to
    know how many files a supplier has attached to a given item and the
    pipeline only needs to know whether each item has *a* reference. Passing
    `session.form` straight to `qualify` therefore reported every submission
    as missing all twelve mandatory documents no matter how many were uploaded.

    Two exclusions, both deliberate:

    - Unreadable uploads do not satisfy an item. They are kept, and the
      supplier is asked for a readable copy, but a scan we could not read is
      not evidence of anything.
    - `auto-fill` is not an item code. It is the marker Smart Document Parsing
      uploads under, so those records must not be mistaken for a mandatory
      attachment.

    Later uploads win on a duplicate code, so replacing a document does not
    leave the previous one counted as a second copy of the same evidence.
    """
    payload = dict(session.form)

    by_code: Dict[str, Dict[str, str]] = {}
    for record in session.documents:
        if record.get("status") == "Unreadable":
            continue
        code = str(record.get("document_code") or "").strip()
        if not code or code.lower() == "auto-fill":
            continue
        reference = str(record.get("reference") or "").strip()
        if not reference:
            continue
        by_code[code] = {
            "document_code": code,
            "attachment_reference": reference,
        }

    # Anything the supplier attached through the form itself is kept, so a
    # submission assembled outside the portal is not stripped of its evidence.
    existing = payload.get("documents")
    for entry in existing if isinstance(existing, (list, tuple)) else []:
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("document_code") or "").strip()
        reference = str(
            entry.get("attachment_reference") or entry.get("reference") or ""
        ).strip()
        if code and reference:
            by_code.setdefault(code, {
                "document_code": code,
                "attachment_reference": reference,
            })

    payload["documents"] = list(by_code.values())
    return payload


def get_dossier(token: str) -> Optional[Dict[str, Any]]:
    """The cached full dossier for a submission, or None if not computed yet."""
    return _load_session(token).internal.get("dossier")


def _clear_cached_dossier(session: Session) -> None:
    """Drop any cached dossier from a session already held in memory.

    Takes the session rather than a token so the caller does not reload it: the
    mutation must land on the same object the caller is about to persist, or
    `_save_session` writes the dossier straight back.
    """
    session.internal.pop("dossier", None)
    session.internal.pop("dossier_computed_at", None)


def clear_dossier(token: str) -> None:
    """Drop a cached dossier so the next report request recomputes it.

    A cached `REJECTED_INCOMPLETE` is the expensive kind to keep: it was
    produced from an incomplete input, and the supplier filling in what was
    missing does not by itself invalidate the session, so the report page would
    keep showing the stale rejection forever.
    """
    session = _load_session(token)
    if session.internal.get("dossier") is not None:
        _clear_cached_dossier(session)
        _save_session(session)


def set_dossier(token: str, dossier: Dict[str, Any]) -> None:
    """Cache the full dossier against the session so it is computed once."""
    session = _load_session(token)
    session.internal["dossier"] = dossier
    # Epoch seconds, matching every other timestamp in this module. The cache
    # is written before this line on purpose: if the stamp itself were to fail,
    # a dossier that cost a 90-second pipeline run would be recomputed on every
    # refresh rather than being thrown away.
    session.internal["dossier_computed_at"] = time.time()
    _save_session(session)


def list_invitations() -> List[Dict[str, Any]]:
    db = _db()
    try:
        _restore()
    except Exception:
        pass
    items = []
    for token, sess in list(_SESSIONS.items()):
        items.append({
            "token": token,
            "vendor_name": sess.vendor_name,
            "supplier_email": sess.supplier_email or "",
            "invited_by": sess.invited_by or "Procurement",
            "created_at": sess.created_at,
            "submitted_at": sess.submitted_at,
            "status": sess.neutral_status,
            "submitted": sess.submitted_at is not None,
            "link": f"{_portal_base()}/supplier/onboarding/{token}",
        })
    items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return items


def delete_invitation(token: str) -> Dict[str, Any]:
    """Withdraw one invitation: the supplier link, the submission, the traces.

    The session row goes first, so from the next request on the supplier's
    token resolves exactly like an unknown one — the portal's 404 says nothing
    either way. What goes with it is everything keyed to that submission:
    the review invites that open it, the compliance alert it raised, the
    approval briefs written against it, its outbox rows (dropped by the caller,
    which owns the notification log) — and, when this was the vendor's last
    live invitation, the Glass Box and Agent Activity rows that name the
    vendor. A vendor with another invitation still open keeps its audit trail,
    because the surviving submission's history points at it.
    """
    session = _find_session(token)
    if session is None:
        raise HTTPException(status_code=404, detail="This link is not valid.")
    vendor = session.vendor_name

    # Asked before this session is popped: with it still counted, "any other
    # session for this vendor" would be true even when this is the only one.
    with _SESSIONS_LOCK:
        vendor_has_other_session = any(
            s.token != token and s.vendor_name == vendor
            for s in _SESSIONS.values()
        )

    alert_key = f"auto-{token[:16]}"

    def _row_vendor(payload: Dict[str, Any]) -> str:
        return str(payload.get("vendor") or payload.get("vendor_name") or "")

    def _doomed(kind: str, key: str, payload: Dict[str, Any]) -> bool:
        if kind == "review_invite":
            return payload.get("submission_token") == token
        if kind == "alert":
            if (
                key == alert_key
                or payload.get("supplier_token") == token
            ):
                return True
        elif kind == "approval":
            if (
                payload.get("supplier_token") == token
                or payload.get("submission_token") == token
            ):
                return True
        elif kind in ("agent_activity", "glassbox"):
            if not vendor_has_other_session and _row_vendor(payload) == vendor:
                return True
        return False

    db = _db()
    if db is not None:
        with _DB_LOCK:
            for kind, key, blob in db.execute(
                "SELECT kind, key, data FROM portal_items"
            ).fetchall():
                try:
                    payload = json.loads(blob)
                except Exception:
                    continue
                if _doomed(kind, key, payload):
                    db.execute(
                        "DELETE FROM portal_items WHERE kind = ? AND key = ?",
                        (kind, key),
                    )
            db.execute("DELETE FROM portal_sessions WHERE token = ?", (token,))
            db.commit()

    # The caches are the live view: rows deleted only on disk would keep
    # showing in the console until the next restart.
    with _SESSIONS_LOCK:
        _SESSIONS.pop(token, None)
    with _ALERTS_LOCK:
        for key in [k for k, a in _ALERTS.items() if _doomed("alert", k, a)]:
            _ALERTS.pop(key, None)
    with _APPROVALS_LOCK:
        for key in [k for k, a in _APPROVALS.items() if _doomed("approval", k, a)]:
            _APPROVALS.pop(key, None)
    if not vendor_has_other_session:
        with _GLASSBOX_LOCK:
            _GLASSBOX[:] = [
                e for e in _GLASSBOX if not _doomed("glassbox", "", e)
            ]
        with _ACTIVITY_LOCK:
            _ACTIVITY[:] = [
                e for e in _ACTIVITY if not _doomed("agent_activity", "", e)
            ]

    return {"token": token, "vendor_name": vendor}


def issue_review_invite(submission_token: str, reviewer_name: str = "", reviewer_email: str = "", invited_by: str = "") -> str:
    """Issue a time-limited token for an internal reviewer to review a submission without login."""
    import secrets
    if not submission_token:
        raise HTTPException(status_code=400, detail="submission_token is required")
    # verify submission exists
    session = _load_session(submission_token)
    if not session:
        raise HTTPException(status_code=404, detail="Submission not found")
    token = secrets.token_urlsafe(32)
    review_key = f"review:{token}"
    payload = {
        "submission_token": submission_token,
        "reviewer_name": reviewer_name,
        "reviewer_email": reviewer_email,
        "invited_by": invited_by,
        "created_at": time.time(),
        "expires_at": time.time() + 7 * 24 * 3600,  # 7 days
    }
    _persist_item("review_invite", review_key, payload)
    return token


def get_review_invite(review_token: str) -> Dict[str, Any]:
    if not review_token:
        raise HTTPException(status_code=400, detail="review_token required")
    item = _get_item("review_invite", f"review:{review_token}")
    if not item:
        raise HTTPException(status_code=404, detail="Review invite not found or expired")
    payload = item.get("payload", {})
    if payload.get("expires_at") and payload["expires_at"] < time.time():
        raise HTTPException(status_code=410, detail="Review invite expired")
    return payload


def review_link(review_token: str) -> str:
    return f"{_portal_base(required=False)}/portal/review/{review_token}"
