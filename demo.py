"""Presenter demo: seeded vendors, a virtual agent clock, autonomous chasers.

Backs the console's presenter toolbar:

    POST /api/demo/run-vendor/{a|b|c}   seed one vendor through the real flow
    POST /api/demo/reset                drop everything the demo seeded
    GET  /api/demo/clock                the agent clock
    POST /api/demo/clock                pause / play / +1 day

The clock is virtual: it sits on real time while paused and moves only while
it runs, so "+1 Day" is a button rather than a wait. Chasers read the virtual
clock, and so do the console's overdue gates (via ``GET /api/demo/clock``),
so one press ages the whole dashboard together — the chaser rows appear in
the outbox and the invitation crosses its overdue threshold in the same
moment.

Nothing here sends mail. Seeds, chasers and escalations are written to the
presenter outbox (``notification_log``) with ``sent=False``, because a demo
button that emails a stranger is a demo button nobody presses twice.

Seeding is not a parallel fixture path: it mints the token with the real
``issue_token``, fills the form, and lets ``run_assessment`` /
``submit_session`` score it. Vendor C's watchlist hit therefore raises its
MLRO alert through the same hook a live submission does.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional

import supplier_portal

# How often the background loop re-checks the chasers. Real seconds.
TICK_SECONDS = 4

# Day thresholds, in virtual days of age, at which the autonomous chaser acts.
CHASER_REMINDER_DAY = 3   # chase the supplier: your form is outstanding
CHASER_ESCALATION_DAY = 7  # escalate to whoever invited them

# The demo suppliers are invited at this address. Default is the test mailbox
# already used across the seed data, so the presenter signed in there can open
# the supplier side of a seeded link as well; override for a different desk.
DEMO_SUPPLIER_EMAIL = "imran.muhammad9100@gmail.com"


# --------------------------------------------------------------------------
# The agent clock
# --------------------------------------------------------------------------

_clock_lock = threading.Lock()
_clock: Dict[str, Any] = {
    # Runs from boot: a paused-at-boot clock falls behind real time the moment
    # the server has been up for a minute, and the console reads this for its
    # ages. Default-running keeps virtual ≈ real until the presenter pauses it.
    "real_anchor": time.time(),
    "virtual_anchor": time.time(),
    "running": True,
}


def _virtual_locked(real_now: Optional[float] = None) -> float:
    """Virtual time at `real_now`. Caller holds `_clock_lock`."""
    elapsed = 0.0
    if _clock["running"]:
        elapsed = (real_now if real_now is not None else time.time()) - _clock["real_anchor"]
    return _clock["virtual_anchor"] + elapsed


def virtual_now() -> float:
    with _clock_lock:
        return _virtual_locked()


def clock_status() -> Dict[str, Any]:
    real = time.time()
    with _clock_lock:
        virtual = _virtual_locked(real)
        running = bool(_clock["running"])
    return {
        "virtual_now": virtual,
        "real_now": real,
        "running": running,
        "offset_seconds": virtual - real,
    }


def set_running(running: bool) -> Dict[str, Any]:
    """Pause or play. The anchors are re-pinned so no time is lost or doubled."""
    with _clock_lock:
        virtual = _virtual_locked()
        _clock["virtual_anchor"] = virtual
        _clock["real_anchor"] = time.time()
        _clock["running"] = bool(running)
    return clock_status()


def advance_days(days: int) -> Dict[str, Any]:
    """Jump the clock forward. Chasers for every crossed threshold fire now."""
    days = max(1, int(days))
    with _clock_lock:
        _clock["virtual_anchor"] = _virtual_locked() + days * 86_400
        _clock["real_anchor"] = time.time()
    return clock_status()


def reset_clock() -> Dict[str, Any]:
    with _clock_lock:
        _clock["virtual_anchor"] = time.time()
        _clock["real_anchor"] = time.time()
        _clock["running"] = True
    return clock_status()


# --------------------------------------------------------------------------
# Autonomous chasers
# --------------------------------------------------------------------------

def _log_chaser(kind: str, subject: str, recipient: str, detail: str,
                token: str, link: str) -> None:
    from main import log_notification
    log_notification(kind, subject, recipient, False, detail, token=token, link=link)


def run_chasers() -> List[Dict[str, str]]:
    """Chase every silent invitation the virtual clock has caught up with.

    Idempotent by construction: each (threshold, token) pair checks the outbox
    for its own row before writing one, so the loop can run every tick and
    "+1 Day" can be pressed repeatedly without stacking duplicate reminders.
    """
    from main import notification_exists

    _ensure_restored()
    virtual = virtual_now()
    fired: List[Dict[str, str]] = []

    with supplier_portal._SESSIONS_LOCK:
        sessions = list(supplier_portal._SESSIONS.values())

    for sess in sessions:
        if sess.submitted_at is not None or not sess.supplier_email:
            continue
        age_days = int((virtual - sess.created_at) // 86_400)
        if age_days < CHASER_REMINDER_DAY:
            continue

        vendor = sess.vendor_name or "A vendor"
        try:
            link = supplier_portal.invitation_link(sess.token)
        except Exception:  # base URL unset — the row still matters
            link = ""

        if age_days >= CHASER_REMINDER_DAY and not notification_exists(
            "chaser-reminder", sess.token
        ):
            subject = f"Reminder: {vendor}'s onboarding form is still outstanding"
            detail = (
                f"Autonomous chaser, day {CHASER_REMINDER_DAY} of the agent clock "
                "— logged, not emailed (demo mode)."
            )
            _log_chaser("chaser-reminder", subject, sess.supplier_email, detail,
                        sess.token, link)
            supplier_portal.log_agent_activity(
                "Supplier Concierge",
                f"Email clarification chase sent to {vendor} "
                f"(day {CHASER_REMINDER_DAY}): onboarding form still outstanding.",
                rule="Spec 3.4; A7",
                vendor=vendor,
                level="L3",
            )
            fired.append({"kind": "chaser-reminder", "subject": subject})

        if age_days >= CHASER_ESCALATION_DAY and not notification_exists(
            "chaser-escalation", sess.token
        ):
            subject = f"Overdue: {vendor} has not responded after 7 days"
            detail = (
                f"Autonomous chaser, day {CHASER_ESCALATION_DAY} — escalation to "
                "the inviter, logged not emailed (demo mode)."
            )
            _log_chaser("chaser-escalation", subject,
                        sess.invited_by or sess.supplier_email, detail,
                        sess.token, link)
            supplier_portal.log_agent_activity(
                "Supplier Concierge",
                f"Escalation chase sent for {vendor} "
                f"(day {CHASER_ESCALATION_DAY}): no response after 7 days.",
                rule="Spec 3.4; A7",
                vendor=vendor,
                level="L3",
            )
            fired.append({"kind": "chaser-escalation", "subject": subject})

    return fired


async def _loop() -> None:
    """While the clock runs, check the chasers on every tick."""
    while True:
        await asyncio.sleep(TICK_SECONDS)
        try:
            if clock_status()["running"]:
                run_chasers()
        except Exception:  # noqa: BLE001 - a chaser must never kill the loop
            pass


_task: Optional[asyncio.Task] = None


def start() -> None:
    """Attach the loop to the running event loop (FastAPI startup)."""
    global _task
    if _task is None or _task.done():
        _task = asyncio.get_running_loop().create_task(_loop())


def stop() -> None:
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
    _task = None


# --------------------------------------------------------------------------
# Seeded vendors
#
# The three recipes were scored against the real engine (build_signals →
# score_factors → weighted_score → tier_for): A lands 1.05, B 1.84, C 2.88 on
# the policy's 1.00–3.00 scale, i.e. exactly the SDD / CDD / EDD split the
# toolbar promises. If a factor's rules change, re-run them before presenting.
# --------------------------------------------------------------------------

def _common_clean_fields() -> Dict[str, Any]:
    return {
        "pep_present": "No",
        "pep_family_member": "No",
        "pep_close_associate": "No",
        "bearer_shares": "No",
        "nominee_shareholders": "No",
        "prior_regulatory_matter": "No",
    }


def _corporate_fields(
    *,
    account_name: str,
    license_no: str,
    trn: str,
    iban: str,
    account_no: str,
    bank: str,
    address: str,
) -> Dict[str, Any]:
    """The entity / TRN / bank / address facts the Vendor file tab renders as
    "Extracted Form Data". Every recipe declares its own so the three demo
    vendors never read as one vendor copy-pasted three times."""
    return {
        "trade_license_no": license_no,
        "trade_license_expiry": "2028-06-30",
        "vat_registration_status": "Registered",
        "vat_registration_no": trn,
        "registered_address": address,
        "bank_name_branch_country": bank,
        "bank_account_name": account_name,
        "bank_account_number": account_no,
        "bank_iban": iban,
        "swift_code": "EBILAEAD",
    }


def _qualification_fields(
    *,
    incorporation: str,
    commenced: str,
    certifications: List[str],
    rep_name: str,
    rep_designation: str,
    rep_id: str,
    rep_id_expiry: str,
    rep_authority: str,
    directors: List[Dict[str, Any]],
    parent: str,
    ubos: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    turnover_y1: str,
    turnover_y2: str,
    turnover_y3: str,
    intermediary: str,
    regulatory: str,
    signatory: str,
    signatory_designation: str,
    signed: str,
) -> Dict[str, Any]:
    """Every remaining mandatory Form NH-PQF-001 answer the qualification gate
    checks: incorporation dates, certifications, the authorised representative,
    ownership and UBO tables, client references, three-year turnover, the
    audited-accounts flag, the FTA disclosure declarations, and the signature
    block.

    Run Vendor seeds the whole form through this, so a demo vendor reaches
    `validation.validate()` complete — Run Qualification must never reject a
    seeded submission as incomplete. The answers that carry the vendor's story
    (C declares an intermediary role and a prior regulatory matter; A and B do
    not) are parameters rather than constants; the rest are the same shape for
    all three.
    """
    return {
        "date_of_incorporation": incorporation,
        "year_of_commencement": commenced,
        "certifications": list(certifications),
        "authorized_representative_name": rep_name,
        "authorized_representative_designation": rep_designation,
        "authorized_representative_id": rep_id,
        "authorized_representative_id_expiry": rep_id_expiry,
        "authorized_representative_authority_basis": rep_authority,
        "directors_and_owners": list(directors),
        "ultimate_parent_company": parent,
        "ubos": list(ubos),
        "client_references": list(references),
        "turnover_year_1": turnover_y1,
        "turnover_year_2": turnover_y2,
        "turnover_year_3": turnover_y3,
        # Item 40a. "Yes" also activates the Item 84 audited-report document,
        # which is why the B and C document packs below carry it.
        "financial_statements_audited": "Yes",
        "address_change_declaration": "No",
        "management_change_declaration": "No",
        "intermediary_declaration": intermediary,
        "subcontracting_declaration": "No",
        "regulatory_history_declaration": regulatory,
        "supplier_declaration": "Yes",
        "authorized_signatory_name": signatory,
        "authorized_signatory_designation": signatory_designation,
        "date_signed": signed,
    }


# Which form fields each seeded document plausibly carried — what Document
# Intelligence "found" in it. Mirrors the match a live upload through
# `store_document` would have made, so the vendor file's per-document
# extraction note tells the same story either way.
_SEEDED_EXTRACTED: Dict[str, List[str]] = {
    "62": ["legal_name", "trade_license_no", "trade_license_expiry"],
    "63": ["vat_registration_no"],
    "64": ["registered_address", "nature_of_business"],
    "67": ["authorized_representative_name", "authorized_representative_id"],
    "68": ["bank_account_name", "bank_account_number", "bank_iban"],
    "69": ["registered_address"],
    "80": ["directors_and_owners"],
}


def _doc_label(code: str) -> str:
    """The form's name for a document item, straight from the validation
    catalog — the same words the checklist and the upload route use."""
    import validation as validation_lib

    for catalog in (
        validation_lib.ALL_SUPPLIER_DOCUMENTS,
        validation_lib.CONDITIONAL_DOCUMENTS,
    ):
        for entry in catalog:
            if entry[0] == code:
                return entry[1]
    return f"Form item {code}"


def _doc_pdf(title: str, subtitle: str) -> bytes:
    """A one-page PDF bearing a document's title — the copy the buyer's
    Vendor file previews for a seeded upload.

    Hand-built instead of library-generated so the demo grows no dependency.
    The xref table is computed from the bytes as they are written, which is
    the part a hand-rolled PDF usually gets wrong: every offset is the length
    of everything before its object.
    """
    def esc(text: str) -> str:
        return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    stream = (
        "BT /F1 18 Tf 56 740 Td (" + esc(title) + ") Tj "
        "0 -28 Td /F1 11 Tf (" + esc(subtitle) + ") Tj "
        "0 -44 Td /F1 9 Tf (Seeded demo copy — previewed by the NH Console "
        "vendor file.) Tj ET"
    ).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: List[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF"
    ).encode("ascii")
    return bytes(out)


DEMO_RECIPES: Dict[str, Dict[str, Any]] = {
    "a": {
        "vendor_name": "Aurora Medical Supplies FZE",
        "tier": "Low Risk (SDD)",
        # Unsubmitted, and aged two days on arrival: the presenter presses
        # "+1 Day" once and the day-3 chaser fires in the outbox.
        "submit": False,
        "age_days": 2,
        "form": {
            **_common_clean_fields(),
            **_corporate_fields(
                account_name="Aurora Medical Supplies FZE",
                license_no="CN-1045786",
                trn="100456789100003",
                iban="AE160331234567890123456",
                account_no="0121000987654321",
                bank="Mashreq Bank, Al Wasl Road Branch, UAE",
                address="Office 405, Wasl Business Centre, Al Wasl Road, Dubai, UAE",
            ),
            **_qualification_fields(
                incorporation="2016-03-14",
                commenced="2016",
                certifications=["iso_9001"],
                rep_name="Sara Al-Mansoori",
                rep_designation="General Manager",
                rep_id="784-1990-1234567-1",
                rep_id_expiry="2029-05-31",
                rep_authority="Power of Attorney attested by the Dubai Notary Public (2024)",
                directors=[
                    {
                        "name": "Sara Al-Mansoori",
                        "position": "General Manager and shareholder",
                        "nationality": "United Arab Emirates",
                    },
                ],
                parent="N/A - independently held",
                ubos=[
                    {
                        "name": "Sara Al-Mansoori",
                        "nationality": "United Arab Emirates",
                        "ownership_percentage": "100",
                    },
                ],
                references=[
                    {
                        "client_name": "Sheikh Khalifa Medical City",
                        "contact_details": "procurement@skmc.ae, +971 2 314 2000",
                    },
                    {
                        "client_name": "Mediclinic City Hospital, Dubai Healthcare City",
                        "contact_details": "supply.chain@mediclinic.ae",
                    },
                    {
                        "client_name": "NMC Royal Hospital, Khalifa City",
                        "contact_details": "tenders@nmc.ae",
                    },
                ],
                turnover_y1="4200000",
                turnover_y2="3850000",
                turnover_y3="3400000",
                intermediary="No",
                regulatory="No",
                signatory="Sara Al-Mansoori",
                signatory_designation="General Manager",
                signed="2026-09-28",
            ),
            "legal_name": "Aurora Medical Supplies FZE",
            "country_of_incorporation": "United Arab Emirates",
            "jurisdiction_risk_tier": "Low",
            "nature_of_business": "Medical consumables trading",
            "goods_services_proposed": "Routine medical consumables and PPE",
            "sensitivity": "Routine",
            "business_role": "Direct supplier",
            "ownership_structure": "Simple",
            "payment_structure": "Standard",
            "estimated_spend_aed": "400000",
            "single_contract_value_aed": "250000",
            "adverse_media_result": "Clear",
        },
    },
    "b": {
        "vendor_name": "Bharat Heavy Fabricators LLC",
        "tier": "Medium Risk (CDD)",
        # Submitted, awaiting the buyer's decision: the Needs-you queue.
        "submit": True,
        "age_days": 0,
        # The Supplier View simulator reads the portal's own document
        # records, so a seeded vendor that never uploaded anything would
        # show an empty checklist. B arrives with a complete pack — including
        # the three documents its own answers activate: Form 73 (spend over
        # AED 375,000), Item 82 (declared ISO certificate) and Item 84
        # (audited statements).
        "documents": [
            ("60", "code-of-conduct"),
            ("61", "confidentiality-agreement"),
            ("62", "trade-licence"),
            ("64", "company-profile"),
            ("66", "board-resolution"),
            ("67", "representative-emirates-id"),
            ("68", "bank-confirmation"),
            ("69", "address-evidence"),
            ("63", "vat-certificate"),
            ("79", "certificate-of-origin"),
            ("80", "ownership-evidence"),
            ("73", "bank-spend-confirmation"),
            ("82", "quality-infosec-certificates"),
            ("84", "audited-balance-sheet"),
        ],
        "form": {
            **_common_clean_fields(),
            **_corporate_fields(
                account_name="Bharat Heavy Fabricators LLC",
                license_no="CN-2098341",
                trn="100789456200003",
                iban="AE950331239876543210987",
                account_no="0121000456789123",
                bank="HSBC Middle East, Jebel Ali Branch, UAE",
                address="Warehouse 12, Jebel Ali Free Zone, Dubai, UAE",
            ),
            **_qualification_fields(
                incorporation="2009-07-22",
                commenced="2009",
                certifications=["iso_9001"],
                rep_name="Rajesh Kumar Nair",
                rep_designation="Managing Director",
                rep_id="Z4567890",
                rep_id_expiry="2031-08-14",
                rep_authority="Board Resolution BR-14/2024 authorising the Managing Director",
                directors=[
                    {
                        "name": "Rajesh Kumar Nair",
                        "position": "Managing Director",
                        "nationality": "India",
                    },
                    {
                        "name": "Anita Sharma",
                        "position": "Director of Finance",
                        "nationality": "India",
                    },
                ],
                parent="Bharat Heavy Investments Pvt Ltd (India, 100%)",
                ubos=[
                    {
                        "name": "Rajesh Kumar Nair",
                        "nationality": "India",
                        "ownership_percentage": "70",
                    },
                    {
                        "name": "Anita Sharma",
                        "nationality": "India",
                        "ownership_percentage": "30",
                    },
                ],
                references=[
                    {
                        "client_name": "Jindal Steel & Power Ltd",
                        "contact_details": "vendor-desk@jindalsteel.in, +91 11 4366 2000",
                    },
                    {
                        "client_name": "Larsen & Toubro FZE, Dubai",
                        "contact_details": "fze.procurement@ltindia.com",
                    },
                    {
                        "client_name": "Gulf Steel Industries LLC, Ajman",
                        "contact_details": "purchase@gulfsteel.ae",
                    },
                ],
                turnover_y1="28500000",
                turnover_y2="24100000",
                turnover_y3="21750000",
                intermediary="No",
                regulatory="No",
                signatory="Rajesh Kumar Nair",
                signatory_designation="Managing Director",
                signed="2026-10-01",
            ),
            "legal_name": "Bharat Heavy Fabricators LLC",
            "country_of_incorporation": "India",
            "jurisdiction_risk_tier": "Medium",
            "nature_of_business": "Industrial steel fabrication",
            "goods_services_proposed": "Steel structures and heavy fabrication",
            "sensitivity": "Moderate",
            "business_role": "Direct",
            "multi_layered_ownership": "Yes",
            # "No": B is a direct supplier to private steel firms (see its
            # client references), and a Yes here would fire the
            # government-facing-intermediary Always-EDD trigger — contradicting
            # both `business_role: Direct` and this recipe's own declared
            # Medium Risk (CDD) tier. The demo's middle vendor is meant to be
            # the one that qualifies.
            "engages_government_officials": "No",
            "payment_structure": "Extended terms",
            "estimated_spend_aed": "1500000",
            "single_contract_value_aed": "900000",
            "adverse_media_result": "Clear",
        },
    },
    "c": {
        "vendor_name": "Caspian Energy Trading Ltd",
        "tier": "High Risk (EDD)",
        # Submitted with an unresolved sanctions match: high tier *and* a
        # watchlist hit, so run_assessment raises the MLRO alert that
        # populates the compliance queue.
        "submit": True,
        "age_days": 0,
        # C's pack answers every document its own answers activate: Form 73
        # (spend over AED 375,000), Item 77 (declared intermediary role),
        # Item 81 (its prior regulatory matter), Item 83 (declared HSE/ESG
        # certification) and Item 84 (audited statements). It still carries no
        # Item 80 ownership evidence — the nominee-holdings story has none,
        # and the goods-supply condition that would demand it is not declared.
        "documents": [
            ("60", "code-of-conduct"),
            ("61", "confidentiality-agreement"),
            ("62", "trade-licence"),
            ("64", "company-profile"),
            ("66", "board-resolution"),
            ("67", "representative-emirates-id"),
            ("68", "bank-confirmation"),
            ("69", "address-evidence"),
            ("63", "vat-certificate"),
            ("79", "certificate-of-origin"),
            ("73", "bank-spend-confirmation"),
            ("77", "intermediary-agreement"),
            ("81", "regulatory-remediation"),
            ("83", "esg-code-of-conduct"),
            ("84", "audited-balance-sheet"),
        ],
        "form": {
            **_corporate_fields(
                account_name="Caspian Energy Trading Ltd",
                license_no="CN-3345620",
                trn="100321654900003",
                iban="AE240331231234567890123",
                account_no="0121000789123456",
                bank="ADCB, Main Branch, Abu Dhabi, UAE",
                address="Level 22, Al Sila Tower, ADGM Square, Abu Dhabi, UAE",
            ),
            **_qualification_fields(
                incorporation="2014-01-19",
                commenced="2014",
                certifications=["hse_esg"],
                rep_name="Aylin Demir",
                rep_designation="Authorized Representative",
                rep_id="U9876543",
                rep_id_expiry="2030-02-28",
                rep_authority="Power of Attorney recorded by the Istanbul Notary (2025)",
                directors=[
                    {
                        "name": "Aylin Demir",
                        "position": "Authorized Representative",
                        "nationality": "Turkiye",
                    },
                    {
                        "name": "Rustam Karimov",
                        "position": "Director",
                        "nationality": "Kazakhstan",
                    },
                ],
                parent="Caspian Energy Holding JSC (Kazakhstan)",
                ubos=[
                    {
                        "name": "Rustam Karimov",
                        "nationality": "Kazakhstan",
                        "ownership_percentage": "60",
                    },
                    {
                        "name": "Aylin Demir",
                        "nationality": "Turkiye",
                        "ownership_percentage": "40",
                    },
                ],
                references=[
                    {
                        "client_name": "Trans-Caspian Trading FZE",
                        "contact_details": "trade@transcaspian.ae",
                    },
                    {
                        "client_name": "Anatolia Energy Import LLC",
                        "contact_details": "desk@anatoliaenergy.com.tr",
                    },
                    {
                        "client_name": "Gulf Petrolink DMCC",
                        "contact_details": "ops@gulfpetrolink.ae",
                    },
                ],
                turnover_y1="96000000",
                turnover_y2="84200000",
                turnover_y3="71500000",
                intermediary="Yes",
                regulatory="Yes",
                signatory="Aylin Demir",
                signatory_designation="Authorized Representative",
                signed="2026-10-02",
            ),
            "legal_name": "Caspian Energy Trading Ltd",
            "country_of_incorporation": "Yemen",
            "jurisdiction_risk_tier": "High",
            "nature_of_business": "Oil and gas brokerage",
            "goods_services_proposed": "Crude oil intermediation",
            "sensitivity": "Sensitive",
            "business_role": "Intermediary",
            "nominee_shareholders": "Yes",
            "acts_as_intermediary": "Yes",
            "payment_structure": "Success fee",
            "estimated_spend_aed": "12000000",
            "single_contract_value_aed": "11000000",
            "adverse_media_result": "Unresolved sanctions",
            "unresolved_sanctions_match": "Yes",
            "pep_present": "Yes",
            "pep_family_member": "No",
            "pep_close_associate": "No",
            "bearer_shares": "No",
            "prior_regulatory_matter": "Yes",
            "prior_compliance_issues": "Yes",
        },
    },
}


_restored_once = False


def _ensure_restored() -> None:
    """Populate the session cache from disk once per process.

    `_restore()` repopulates the cache on first use of the store; without it a
    reset pressed right after a (re)load — or the first chaser tick — scans an
    empty dict while the rows sit on disk. Once is enough: every later write
    lands in the cache directly, which is the store's own read-through design.
    """
    global _restored_once
    if _restored_once:
        return
    try:
        supplier_portal._db()
        supplier_portal._restore()
        _restored_once = True
    except Exception:  # noqa: BLE001 - memory-only store restores as a no-op
        pass


def _demo_tokens(which: Optional[str] = None) -> List[str]:
    _ensure_restored()
    with supplier_portal._SESSIONS_LOCK:
        return [
            token
            for token, sess in supplier_portal._SESSIONS.items()
            if sess.internal.get("demo") and (which is None or sess.internal.get("demo") == which)
        ]


def _drop_demo_sessions(which: Optional[str] = None) -> List[str]:
    """Forget seeded sessions everywhere they are stored. Returns their tokens."""
    tokens = _demo_tokens(which)
    if not tokens:
        return []

    from main import delete_notifications

    # Alerts raised by these sessions (Vendor C's) go with them.
    import supplier_portal as sp

    with sp._ALERTS_LOCK:
        alert_ids = [
            alert_id
            for alert_id, alert in sp._ALERTS.items()
            if alert.get("supplier_token") in tokens
        ]
        for alert_id in alert_ids:
            sp._ALERTS.pop(alert_id, None)

    with sp._SESSIONS_LOCK:
        for token in tokens:
            sp._SESSIONS.pop(token, None)

    db = sp._db()
    if db is not None:
        with sp._DB_LOCK:
            for token in tokens:
                db.execute("DELETE FROM portal_sessions WHERE token = ?", (token,))
            for alert_id in alert_ids:
                db.execute(
                    "DELETE FROM portal_items WHERE kind = 'alert' AND key = ?",
                    (alert_id,),
                )
            db.commit()

    try:
        delete_notifications(tokens)
    except Exception:  # noqa: BLE001 - outbox hygiene must not fail a reset
        pass
    return tokens


def run_vendor(which: str, inviter: str = "") -> Dict[str, Any]:
    """Seed one vendor through the real invitation → (fill → submit) flow."""
    from fastapi import HTTPException

    from main import log_notification

    which = (which or "").strip().lower()
    recipe = DEMO_RECIPES.get(which)
    if recipe is None:
        raise HTTPException(
            status_code=400,
            detail="Unknown vendor. Use a, b or c.",
        )

    # Re-running a vendor replaces its previous run: two Vendor A rows in the
    # presenter's dashboard is one Vendor A too many.
    _drop_demo_sessions(which)

    token = supplier_portal.issue_token(
        recipe["vendor_name"],
        invited_by=inviter or "presenter",
        supplier_email=_supplier_email(),
    )
    session = supplier_portal._load_session(token)
    session.form.update(recipe["form"])
    session.internal = dict(session.internal, demo=which)
    # Seed the portal's document records the recipe asks for, in the same
    # `code-code-slug.pdf` shape store_document writes. Vendor A has no list
    # (it has not submitted yet); B and C carry every document their own
    # answers activate, so the completeness gate passes on a seeded run.
    # Each record keeps a real preview copy — the buyer's Vendor file serves
    # these bytes exactly like a live upload's, so the demo is not a column
    # of dead links.
    for code, slug in recipe.get("documents", ()):
        reference = f"{code}-{code}-{slug}.pdf"
        record = {
            "reference": reference,
            "document_code": code,
            "status": "Read",
            "extracted_fields": list(_SEEDED_EXTRACTED.get(code, ())),
        }
        supplier_portal._attach_preview(
            record,
            _doc_pdf(_doc_label(code), recipe["vendor_name"]),
            reference,
        )
        session.documents.append(record)
    if recipe.get("age_days"):
        session.created_at = time.time() - float(recipe["age_days"]) * 86_400
    supplier_portal._save_session(session)

    try:
        link = supplier_portal.invitation_link(token)
    except Exception:  # noqa: BLE001
        link = ""
    log_notification(
        "invitation",
        f"Invitation created: {recipe['vendor_name']}",
        session.supplier_email,
        False,
        "Presenter demo seed — logged, not emailed (demo mode).",
        token=token,
        link=link,
    )
    supplier_portal.log_agent_activity(
        "Intake and Triage",
        f"Invitation emailed to {recipe['vendor_name']} with a secure link.",
        rule="Spec 3.4 day 0",
        vendor=recipe["vendor_name"],
        level="L3",
    )

    if recipe.get("submit"):
        # Scores it, moves it to "Under review" and — for Vendor C — raises
        # the MLRO alert, all through the live code path. notify=False keeps
        # the seed from mailing the buyer on every toolbar press.
        supplier_portal.submit_session(token, notify=False)
    else:
        # Score it so the dossier exists internally, but leave it unsubmitted:
        # this is the vendor the autonomous chaser chases.
        supplier_portal.run_assessment(session)
        supplier_portal._save_session(session)

    seeded_docs = recipe.get("documents") or ()
    if seeded_docs:
        # What the background agents did with that pack: one Document
        # Intelligence entry per seeded vendor, so the Vendor file's audit
        # log carries the AI findings beside the Glass Box's clearance trace.
        supplier_portal.log_agent_activity(
            "Document Intelligence",
            f"Read {len(seeded_docs)} documents for {recipe['vendor_name']}: "
            f"items {', '.join(code for code, _ in seeded_docs)}.",
            rule="Document checklist; Items 60-80",
            vendor=recipe["vendor_name"],
            level="L3",
        )

    fresh = supplier_portal._load_session(token)
    alert_raised = any(
        alert.get("supplier_token") == token
        for alert in _alerts_snapshot()
    )
    return {
        "which": which,
        "vendor": recipe["vendor_name"],
        "token": token,
        "link": link,
        "status": fresh.neutral_status,
        "submitted": fresh.submitted_at is not None,
        "tier": fresh.internal.get("assigned_risk_tier") or recipe["tier"],
        "expected_tier": recipe["tier"],
        "alert_raised": alert_raised,
    }


def _alerts_snapshot() -> List[Dict[str, Any]]:
    import supplier_portal as sp

    with sp._ALERTS_LOCK:
        return list(sp._ALERTS.values())


def reset_demo() -> Dict[str, Any]:
    """Drop every seeded vendor, their alerts, their outbox rows and chasers."""
    from main import delete_chaser_notifications

    tokens = _drop_demo_sessions(None)
    try:
        delete_chaser_notifications()
    except Exception:  # noqa: BLE001
        pass
    clock = reset_clock()
    return {"dropped_tokens": len(tokens), "clock": clock}


def _supplier_email() -> str:
    import os

    return (os.environ.get("DEMO_SUPPLIER_EMAIL", "").strip()
            or DEMO_SUPPLIER_EMAIL)
