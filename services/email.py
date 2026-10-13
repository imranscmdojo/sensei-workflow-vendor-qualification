"""
Outbound email for the onboarding portal.

Deliberately small and synchronous-safe: one function, one template, one SMTP
session. Anything richer belongs in a queue, and there is no queue yet.

**Unconfigured is not an error.** `send()` returns a result saying so instead of
raising, because a supplier's submission must still be recorded and scored if
SMTP happens to be down. Failing the submit because a mail server is unreachable
would lose the vendor's data to an outage in a system that has no business being
in that request path. The buyer-visible consequence is a logged warning and a
`submitted_no_notification` result, which the caller surfaces.

Configuration, all optional:

    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_USE_TLS, SMTP_FROM,
    NOTIFICATION_TO

`SMTP_USERNAME`/`SMTP_PASSWORD` are accepted as aliases because that is the
generic spelling, but this repo's other backends already use `SMTP_USER` and
`SMTP_PASS` and one convention across the estate is worth more than the generic
name.

`NOTIFICATION_TO` overrides the per-invitation recipient and exists so a pilot
can route everything to one mailbox before real recipients exist.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional

SUBJECT = "Vendor submitted their onboarding form"


@dataclass
class SendResult:
    sent: bool
    detail: str
    recipient: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"sent": self.sent, "detail": self.detail, "recipient": self.recipient}


def configured() -> bool:
    return bool(os.environ.get("SMTP_HOST", "").strip())


def submission_body(
    vendor_name: str,
    tier: str,
    score: str,
    due_diligence: str,
    link: str,
) -> str:
    """Plain-text alternative. Kept adjacent to the HTML so the two cannot drift."""
    return (
        f"{vendor_name} has submitted their vendor onboarding form.\n\n"
        f"  Risk tier      {tier}\n"
        f"  Score          {score}\n"
        f"  Due diligence  {due_diligence}\n\n"
        f"Review it here:\n{link}\n\n"
        "The supplier sees only 'Under review'. The tier, score and any screening\n"
        "result above are internal and are not shown to them.\n"
    )


def submission_html(
    vendor_name: str,
    tier: str,
    score: str,
    due_diligence: str,
    link: str,
) -> str:
    esc = (
        vendor_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f"""<!doctype html>
<html><body style="margin:0;padding:24px;background:#f6f6f4;
  font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1c1c1a;">
  <div style="max-width:560px;margin:0 auto;background:#fff;border:1px solid #e3e3df;
    border-radius:12px;padding:26px 28px;">
    <div style="text-align:center;padding-bottom:16px;border-bottom:1px solid #eeeeea;margin-bottom:16px;"><img src="cid:nh-logo" alt="National Holding" width="180" style="display:inline-block;max-width:100%;height:auto;"></div>
    <p style="margin:0 0 4px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;
      color:#6b6b66;">Vendor onboarding</p>
    <h1 style="margin:0 0 16px;font-size:20px;line-height:1.3;">{esc} submitted their form</h1>
    <table style="width:100%;border-collapse:collapse;margin:0 0 20px;font-size:14px;">
      <tr><td style="padding:8px 0;border-top:1px solid #eee;color:#6b6b66;">Risk tier</td>
        <td style="padding:8px 0;border-top:1px solid #eee;font-weight:600;">{tier}</td></tr>
      <tr><td style="padding:8px 0;border-top:1px solid #eee;color:#6b6b66;">Score</td>
        <td style="padding:8px 0;border-top:1px solid #eee;font-weight:600;">{score}</td></tr>
      <tr><td style="padding:8px 0;border-top:1px solid #eee;color:#6b6b66;">Due diligence</td>
        <td style="padding:8px 0;border-top:1px solid #eee;font-weight:600;">{due_diligence}</td></tr>
    </table>
    <a href="{link}" style="display:inline-block;background:#1c1c1a;color:#fff;
      text-decoration:none;padding:11px 20px;border-radius:8px;font-size:14px;
      font-weight:600;">Review submission</a>
    <p style="margin:22px 0 0;font-size:12px;line-height:1.6;color:#6b6b66;">
      The supplier sees only &ldquo;Under review&rdquo;. The tier, score and any screening
      result above are internal and are not shown to them.</p>
  </div>
</body></html>"""


def send(
    to: str,
    subject: str,
    text: str,
    html: str,
    use_override: bool = True,
) -> SendResult:
    """Send one message. Never raises.

    A notification failure must not take down the request that triggered it, so
    every transport error is caught and returned. The caller decides what a
    `sent=False` means for its own flow.
    """
    # NOTIFICATION_TO aims every message at one pilot inbox, which is right for
    # internal mail and catastrophic for a supplier's: an approval addressed to
    # the pilot means the supplier is never told. Callers that mail the outside
    # world pass use_override=False and keep the real recipient.
    if use_override:
        recipient = (
            (os.environ.get("NOTIFICATION_TO", "") or "").strip() or (to or "").strip()
        )
    else:
        recipient = (to or "").strip()
    if not recipient:
        return SendResult(
            False,
            (
                "No recipient: no NOTIFICATION_TO and no inviter on file."
                if use_override
                else "No recipient on file for this supplier."
            ),
        )
    if not configured():
        return SendResult(
            False, "SMTP is not configured (SMTP_HOST is unset); notification skipped.",
            recipient,
        )

    host = os.environ["SMTP_HOST"].strip()
    port = int(os.environ.get("SMTP_PORT", "587"))
    sender = (
        os.environ.get("SMTP_FROM")
        or os.environ.get("SMTP_USER")
        or os.environ.get("SMTP_USERNAME")
        or ""
    ).strip()
    if not sender:
        return SendResult(False, "SMTP_FROM is unset, so there is no envelope sender.",
                          recipient)

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    _attach_logo(message, html)

    try:
        if os.environ.get("SMTP_USE_TLS", "1") not in ("0", "false", "False"):
            # Port 465 is implicit TLS and cannot be STARTTLS'd; guessing wrong
            # here produces an opaque handshake failure.
            if port == 465:
                with smtplib.SMTP_SSL(host, port, timeout=20,
                                      context=ssl.create_default_context()) as smtp:
                    _authenticate(smtp)
                    smtp.send_message(message)
            else:
                with smtplib.SMTP(host, port, timeout=20) as smtp:
                    smtp.starttls(context=ssl.create_default_context())
                    _authenticate(smtp)
                    smtp.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=20) as smtp:
                _authenticate(smtp)
                smtp.send_message(message)
    except Exception as exc:  # noqa: BLE001 - any transport failure is a no-send
        return SendResult(False, f"{type(exc).__name__}: {exc}".strip(), recipient)

    return SendResult(True, "sent", recipient)


def _authenticate(smtp: smtplib.SMTP) -> None:
    user = os.environ.get("SMTP_USER") or os.environ.get("SMTP_USERNAME") or ""
    password = os.environ.get("SMTP_PASS") or os.environ.get("SMTP_PASSWORD") or ""
    user = user.strip()
    if user:
        smtp.login(user, password)


def notify_submission(
    to: Optional[str],
    vendor_name: str,
    tier: str,
    score: str,
    due_diligence: str,
    link: str,
) -> SendResult:
    """The one notification this system sends.

    It goes to the inviter — the buyer who invited the supplier — because that
    is who has to review the submission. `NOTIFICATION_TO` (the pilot inbox) is
    a fallback for when no inviter is on file, never an override: aiming the
    buyer's own submission notice at the pilot meant the buyer never learned
    their vendor submitted while the pilot saw "sent".
    """
    recipient = ((to or "").strip()
                 or (os.environ.get("NOTIFICATION_TO", "") or "").strip())
    return send(
        recipient,
        f"{SUBJECT}: {vendor_name}",
        submission_body(vendor_name, tier, score, due_diligence, link),
        submission_html(vendor_name, tier, score, due_diligence, link),
        use_override=False,
    )

# --------------------------------------------------------------------------
# Supplier-facing messages
# --------------------------------------------------------------------------
#
# These are the two messages a supplier receives, and they are the only mail in
# this system addressed to the outside. They carry the status and nothing else:
# no tier, no score, no screening result, no due-diligence level.
#
# The firewall in `supplier_portal` governs what the supplier's browser can
# fetch. It cannot govern what arrives in their inbox, so the rule is enforced
# twice, deliberately — once in the API and once here — because a rejection email
# that said "High Risk (EDD)" would disclose the entire assessment through the
# one channel the supplier was never meant to see it in.


def invitation_body(vendor_name: str, link: str) -> str:
    """Plain-text invitation. Kept adjacent to the HTML so the two cannot drift."""
    return (
        f"Hello,\n\n"
        f"You have been invited to complete a vendor onboarding form"
        f"{f' for {vendor_name}' if vendor_name else ''}.\n\n"
        f"Open your form:\n{link}\n\n"
        f"You can save and return to it as often as you like — nothing is sent to\n"
        f"the reviewer until you press submit. If the link stops working, ask your\n"
        f"buyer to send a new one; it is tied to this submission and nothing else.\n\n"
        f"You will get an email at this address when your submission is reviewed.\n"
    )


def invitation_html(vendor_name: str, link: str) -> str:
    esc = _escape(vendor_name)
    who = f" for <strong>{esc}</strong>" if vendor_name else ""
    return _shell(
        eyebrow="Vendor onboarding",
        heading=f"You have been invited{who}",
        paragraphs=[
            "Please complete and submit your vendor onboarding form. You can save "
            "and return to it as often as you like — nothing reaches the reviewer "
            "until you press submit.",
            "If the link stops working, ask your buyer to send a new one. It is "
            "tied to this submission and nothing else.",
            "You will get an email at this address when your submission is reviewed.",
        ],
        cta="Open your form",
        link=link,
    )


def decision_body(
    vendor_name: str,
    approved: bool,
    reason: str,
    link: str,
) -> str:
    """Plain-text outcome notice."""
    heading = (
        "Your vendor onboarding submission has been approved"
        if approved
        else "Your vendor onboarding submission needs changes"
    )
    lines = [
        f"Hello,\n\n",
        f"{heading}"
        + (f" for {vendor_name}" if vendor_name else "")
        + ".\n\n",
    ]
    if not approved:
        # A reason is not always available. The approval-brief route records an
        # outcome without a justification field, and printing "the following
        # changes:" above an empty line reads as a bug to the supplier. Say the
        # same thing without the empty quote in that case.
        if (reason or "").strip():
            lines.append(
                "The reviewer has asked for the following changes:\n\n"
                f"  {reason}\n\n"
                "Open your form, make those changes and submit again:\n"
                f"{link}\n\n"
            )
        else:
            lines.append(
                "Open your form to see what needs changing, then submit it "
                f"again:\n{link}\n\n"
            )
    else:
        lines.append(
            "There is nothing further for you to do. Your form and documents are\n"
            "on file with the reviewer.\n\n"
        )
    lines.append(
        "You can reopen your form at any time to check its current status:\n"
        f"{link}\n"
    )
    return "".join(lines)


def decision_html(
    vendor_name: str,
    approved: bool,
    reason: str,
    link: str,
) -> str:
    esc = _escape(vendor_name)
    who = f" for <strong>{esc}</strong>" if vendor_name else ""
    if approved:
        paragraphs = (
            ["There is nothing further for you to do. Your form and documents are on "
             "file with the reviewer."]
        )
    elif (reason or "").strip():
        paragraphs = [
            "The reviewer has asked for the following changes:",
            f"<em>{_escape(reason)}</em>",
            "Open your form, make those changes and submit again.",
        ]
    else:
        # Same gap as the text half: a decision recorded without a justification
        # must not render an empty quote block.
        paragraphs = ["Open your form to see what needs changing, then submit it again."]
    return _shell(
        eyebrow="Vendor onboarding",
        heading=(
            "Your submission has been approved" if approved
            else "Your submission needs changes"
        ) + who,
        paragraphs=paragraphs,
        cta="Check status" if approved else "Open your form",
        link=link,
        accent="#0f7b3d" if approved else "#8a5a00",
    )


def notify_supplier_invitation(to: str, vendor_name: str, link: str) -> SendResult:
    """The invitation itself."""
    return send(
        to,
        f"{SUBJECT.split(':')[0]}: please complete your onboarding form",
        invitation_body(vendor_name, link),
        invitation_html(vendor_name, link),
        use_override=False,
    )


def submission_confirmation_body(vendor_name: str, link: str) -> str:
    """Plain-text receipt for the supplier's own submission."""
    who = f" for {vendor_name}" if vendor_name else ""
    return (
        "Hello,\n\n"
        f"We have received your vendor onboarding submission{who}.\n\n"
        "It is now with the reviewer. You do not need to do anything else — "
        "if anything needs changing, the reviewer will come back to you.\n\n"
        "Check its status at any time:\n"
        f"{link}\n"
    )


def submission_confirmation_html(vendor_name: str, link: str) -> str:
    """The receipt as the supplier sees it: the card, with a status button."""
    esc = _escape(vendor_name)
    who = f" for <strong>{esc}</strong>" if vendor_name else ""
    return _shell(
        eyebrow="Vendor onboarding",
        heading=f"We have received your submission{who}",
        paragraphs=[
            "It is now with the reviewer. There is nothing further for you to do "
            "right now — if anything needs changing, the reviewer will contact "
            "you.",
            "You can come back at any time to see where your submission stands.",
        ],
        cta="View status",
        link=link,
        accent="#0f7b3d",
    )


def notify_supplier_submission_received(
    to: str, vendor_name: str, link: str
) -> SendResult:
    """The supplier's receipt, sent the moment they submit.

    Separate from the buyer's submission notice, which reports the assessment
    to the inviter. This one confirms only that the form landed and points at
    the supplier's own status page — no tier, no score, nothing the neutral
    firewall keeps from them.
    """
    return send(
        to,
        "Your vendor onboarding submission has been received"
        + (f" — {vendor_name}" if vendor_name else ""),
        submission_confirmation_body(vendor_name, link),
        submission_confirmation_html(vendor_name, link),
        use_override=False,
    )


def status_change_body(vendor_name: str, status: str, link: str) -> str:
    """Plain-text notice that the supplier's status moved.

    Carries the status word and nothing else. The MLRO's justification and the
    re-check's internal note are both buyer-side reasoning, and both would break
    the neutral firewall if they travelled — so this template has no `reason`
    parameter at all, rather than one the caller has to remember to leave empty.
    """
    who = f" for {vendor_name}" if vendor_name else ""
    if status == "Action required":
        lead = (
            "Your vendor onboarding submission needs your attention"
            f"{who}.\n\n"
            "Open your form to see what is being asked, then update and "
            "resubmit it.\n\n"
        )
    else:
        lead = (
            "Your vendor onboarding submission is now under review"
            f"{who}.\n\n"
            "There is nothing further for you to do right now. We will contact "
            "you if anything is needed.\n\n"
        )
    return (
        "Hello,\n\n"
        f"{lead}"
        "Check its status at any time:\n"
        f"{link}\n"
    )


def status_change_html(vendor_name: str, status: str, link: str) -> str:
    """The status notice as the supplier sees it, with a status button."""
    esc = _escape(vendor_name)
    who = f" for <strong>{esc}</strong>" if vendor_name else ""
    action = status == "Action required"
    return _shell(
        eyebrow="Vendor onboarding",
        heading=(
            "Your submission needs your attention" if action
            else "Your submission is now under review"
        ) + who,
        paragraphs=(
            [
                "Open your form to see what is being asked, then update and "
                "resubmit it.",
            ]
            if action
            else [
                "There is nothing further for you to do right now. We will "
                "contact you if anything is needed.",
            ]
        ),
        cta="Open your form" if action else "View status",
        link=link,
        accent="#8a5a00" if action else "#0f7b3d",
    )


def notify_supplier_status_change(
    to: str, vendor_name: str, status: str, link: str
) -> SendResult:
    """Tell the supplier their status moved, and nothing about why.

    Two callers: the MLRO's alert decision, and the document re-check that
    moves a sent-back submission forward again. Both change `neutral_status`
    silently today, which leaves a supplier staring at a page whose status
    word flipped with no mail explaining that it did.
    """
    verb = "needs your attention" if status == "Action required" else "is under review"
    return send(
        to,
        f"Your vendor onboarding submission {verb}"
        + (f" — {vendor_name}" if vendor_name else ""),
        status_change_body(vendor_name, status, link),
        status_change_html(vendor_name, status, link),
        use_override=False,
    )


def notify_supplier_decision(
    to: str,
    vendor_name: str,
    approved: bool,
    reason: str,
    link: str,
) -> SendResult:
    """The outcome. Carries the status and the reason, never the assessment."""
    return send(
        to,
        (
            "Your vendor onboarding submission has been approved"
            if approved
            else "Your vendor onboarding submission needs changes"
        ) + (f" — {vendor_name}" if vendor_name else ""),
        decision_body(vendor_name, approved, reason, link),
        decision_html(vendor_name, approved, reason, link),
        use_override=False,
    )


def _logo_png() -> Optional[bytes]:
    """The National Holding mark, shipped beside this module.

    Embedded in the message (see `_attach_logo`) rather than referenced by
    URL: the pilot inbox is real mail and mail clients cannot reach
    localhost, so a hosted link would render as a broken image everywhere
    that matters. A CID part travels inside the message itself.
    """
    try:
        with open(
            os.path.join(os.path.dirname(__file__), "nh-logo.png"), "rb"
        ) as fh:
            return fh.read()
    except OSError:
        return None


def _attach_logo(message: EmailMessage, html: str) -> None:
    """Wrap the HTML part in a multipart/related with the inline logo.

    `EmailMessage.add_related()` refuses to convert an alternative (CPython
    raises "Cannot convert alternative to related"), so the nesting is done
    by hand, producing the canonical layout:

        multipart/alternative
        ├── text/plain               (first choice: no HTML, no images)
        └── multipart/related
            ├── text/html            (the root part: references cid:nh-logo)
            └── image/png            (Content-ID <nh-logo>, inline)

    Never raises — a missing file or a changed API must not stop the mail.
    """
    if "cid:nh-logo" not in html:
        return
    logo = _logo_png()
    if not logo:
        return
    try:
        payload = message.get_payload()
        if not isinstance(payload, list) or len(payload) != 2:
            return  # not the [plain, html] this template builds — leave it
        image = EmailMessage(policy=message.policy)
        image.set_content(
            logo,
            maintype="image",
            subtype="png",
            disposition="inline",
            cid="<nh-logo>",
        )
        related = MIMEMultipart("related")
        related.attach(payload[1])  # text/html becomes the root part
        related.attach(image)
        payload[1] = related  # payload is the live list, not a copy
    except Exception:  # noqa: BLE001 - presentation must not block delivery
        pass


def _escape(value: str) -> str:
    return (
        (value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _shell(
    *,
    eyebrow: str,
    heading: str,
    paragraphs: list[str],
    cta: str,
    link: str,
    accent: str = "#1c1c1a",
) -> str:
    """The shared frame, so supplier mail cannot drift from buyer mail.

    Only accepts pre-escaped values in `heading` and `paragraphs`, which is why
    the callers above escape before interpolating. Getting that wrong would put
    a supplier's company name into the HTML as markup.
    """
    body = "".join(
        f'<p style="margin:0 0 14px;font-size:14px;line-height:1.65;color:#3d3d38;">'
        f"{para}</p>"
        for para in paragraphs
    )
    # The National Holding mark leads the card, above the eyebrow. The
    # cid: reference is paired with an inline part attached in `send()` —
    # self-contained, no hosted URL needed.
    logo = (
        '<div style="text-align:center;padding-bottom:16px;'
        'border-bottom:1px solid #eeeeea;margin-bottom:16px;">'
        '<img src="cid:nh-logo" alt="National Holding" width="180" '
        'style="display:inline-block;max-width:100%;height:auto;"></div>'
    )
    return f"""<!doctype html>
<html><body style="margin:0;padding:24px;background:#f6f6f4;
  font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1c1c1a;">
  <div style="max-width:560px;margin:0 auto;background:#fff;border:1px solid #e3e3df;
    border-radius:12px;padding:26px 28px;">
    {logo}
    <p style="margin:0 0 4px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;
      color:#6b6b66;">{eyebrow}</p>
    <h1 style="margin:0 0 16px;font-size:20px;line-height:1.3;">{heading}</h1>
    {body}
    <a href="{link}" style="display:inline-block;background:{accent};color:#fff;
      text-decoration:none;padding:11px 20px;border-radius:8px;font-size:14px;
      font-weight:600;">{cta}</a>
  </div>
</body></html>"""
