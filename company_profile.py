"""
Smart auto-fill field mapping — Tool 1.

Turns the text of a Trade Licence or Company Profile into pre-filled form fields.

Why this is rule-based and not a model call
------------------------------------------
A trade licence and a company profile are label/value documents: "Licence number
CN-1094821", "Expiry date 11 April 2027", "TRN 100293847500003". The labels are
stable, the values sit after them, and a wrong guess is visible to the officer who
reviews it before submitting. A model would cost a request on every upload to
produce something less predictable and less auditable, and would turn a purely
local operation into one that fails when the model quota is exhausted. So the
mapping is a table of label patterns, and anything it cannot place is simply left
empty for the officer to type.

That "left empty" is the important property. This module never guesses a field
from weak evidence, and it never returns a value it is not confident about. Every
value it does return carries the source line it came from, so the UI can show the
officer exactly what was read and let them correct it. Auto-fill is a
head start, not an authority.

The extracted keys are `VendorForm` field names, so the client can assign them
straight onto the form.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# `exact`   — a label on the document matched directly.
# `inferred` — derived rather than read verbatim (country from the address, the
#              signatory from a name/role pair, VAT status from the TRN's
#              presence). Always shown to the officer as inferred.
EXACT = "exact"
INFERRED = "inferred"


@dataclass
class ExtractedField:
    """One proposed value, with enough context to check it."""

    field: str
    value: str
    confidence: str
    #: The label the officer sees in the form.
    label: str
    #: The document line this was read from, verbatim.
    source: str
    #: Which wizard step the field lives on, so the UI can send them there.
    step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "confidence": self.confidence,
            "label": self.label,
            "source": self.source,
            "step": self.step,
        }


# --------------------------------------------------------------------------
# Field catalogue
# --------------------------------------------------------------------------
# step 0 = Company details / representative, step 3 = Bank & commercial terms.
# Keys are VendorForm fields; the order of this dict is the order the client
# renders the review list in, so the things a licence always states come first.

FIELD_LABELS: Dict[str, str] = {
    "legal_name": "Company / legal name",
    "trade_license_no": "Trade licence number",
    "trade_license_expiry": "Trade licence expiry date",
    "registered_address": "Registered address",
    "country_of_incorporation": "Country of incorporation",
    "date_of_incorporation": "Date of incorporation",
    "year_of_commencement": "Year of commencement of business",
    "vat_registration_status": "VAT registration status",
    "vat_registration_no": "TRN VAT number",
    "authorized_representative_name": "Representative name",
    "authorized_representative_designation": "Designation / capacity",
    "compliance_contact": "Primary compliance contact",
    "bank_name_branch_country": "Bank name / branch / country",
    "bank_account_name": "Bank account name",
    "bank_account_number": "Bank account number",
    "bank_iban": "Bank IBAN",
}

FIELD_STEP: Dict[str, int] = {
    "bank_name_branch_country": 3,
    "bank_account_name": 3,
    "bank_account_number": 3,
    "bank_iban": 3,
}

# Label patterns per field, matched case-insensitively at the start of a line.
# Longer patterns are tried first so "trade licence number" wins over "licence".
# Note what is deliberately absent: a bare "date" label. Documents put "Date:"
# next to unrelated values (the signature block, the issue date), and matching it
# would overwrite real dates with the document's print date.
FIELD_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "legal_name": (
        # The slash forms come first because they are how these labels are
        # printed on a real form. "Company / legal name Apex Gulf …" matched
        # only "^company" without them, leaving "/ legal name Apex Gulf …".
        "company / legal name", "legal name / company name",
        "legal name", "legal entity name", "registered name",
        "name of company", "company name", "company", "licensee", "licence holder",
        "license holder", "taxable person",
    ),
    "trade_license_no": (
        "trade licence number", "trade license number",
        "trade licence no", "trade license no",
        "licence number", "license number", "licence no", "license no",
        "licence registration number", "commercial licence number",
        "commercial license number", "establishment number",
    ),
    "trade_license_expiry": (
        "trade licence expiry", "trade license expiry",
        "licence expiry date", "license expiry date",
        "licence expiry", "license expiry",
        "expiry date", "expiration date", "expires on", "valid until",
        "valid till", "valid up to",
    ),
    "registered_address": (
        "registered address", "registered office address", "principal address",
        "office address", "business address", "address",
    ),
    "country_of_incorporation": (
        "country of incorporation", "country of registration",
        "country", "jurisdiction of incorporation",
    ),
    "date_of_incorporation": (
        # Deliberately narrow. "Registration date" on a VAT certificate is the
        # *tax* registration date, and reading that into a field called date of
        # incorporation would put a different fact into a compliance field. A
        # document that never says "incorporation" leaves this empty.
        "date of incorporation", "incorporation date", "incorporated on",
        "date incorporated",
    ),
    "year_of_commencement": (
        # Longest first. On "Year of commencement of business 2015" the short
        # label leaves the value as "of business 2015".
        "year of commencement of business", "year of commencement",
        "commencement year", "year established", "year of establishment",
        "year founded", "established in", "since",
    ),
    "vat_registration_no": (
        "trn", "trn number", "trn vat number", "tax registration number",
        "vat registration number", "vat number", "vat registration no",
        "tax number",
    ),
    "authorized_representative_name": (
        "authorised representative", "authorized representative",
        "name of representative", "representative name", "signatory name",
    ),
    "authorized_representative_designation": (
        "designation", "capacity", "position", "title",
        "authorised signatory", "authorized signatory",
    ),
    "compliance_contact": (
        "compliance contact", "compliance email", "contact email", "email",
    ),
    "bank_name_branch_country": (
        # See the note on legal_name: the slash form has to be matched whole.
        "bank name / branch / country", "bank / branch / country",
        "bank and branch", "bank name and branch", "bank name", "bank and branch country",
        "issuing bank", "name of bank", "beneficiary bank",
    ),
    "bank_account_name": (
        "account name", "account holder", "name of account holder",
        "beneficiary name",
    ),
    "bank_account_number": (
        # No "iban" here. An IBAN is the field's own value, and listing it as an
        # account-number alias copied it into bank_account_number as well.
        "account number", "account no",
    ),
    "bank_iban": (
        "international bank account number",
    ),
}

# Words that may precede a catalogue label without changing what the label
# means. Closed on purpose: every one of these has been seen in front of a real
# label ("Primary compliance contact", "Bank account name", "Legal entity
# name"), and the set is small enough that a stray word in a sentence will not
# turn a line into a labelled value.
_LABEL_QUALIFIER = (
    r"(?:primary|main|bank|supplier|vendor|company|legal|registered|"
    r"authoris(?:ed)?|authorized|tax|vat|group|official)"
    r"\s+"
)

# TRN and IBAN are matched as value patterns rather than through the label table.
# Their labels are single words that these documents print without a colon
# ("TRN 100293847500003", "IBAN AE07 0331 …"), and requiring a colon would miss
# them; allowing a space would let a stray "TRN" in body text match anything.
_TRN_LABEL_RE = re.compile(
    r"\b(?:TRN|Tax\s+Registration\s+Number|VAT\s+Registration\s+(?:No\.?|Number))"
    r"\s*[::\-]?\s*(\d[\d \t]{10,24})",
    re.IGNORECASE,
)
_IBAN_LABEL_RE = re.compile(
    r"\b(?:IBAN|International\s+Bank\s+Account\s+Number)"
    # A literal space, never \s: \s matches the newline, so a greedy value group
    # ran past the end of the line into the next label ("…0123 456\nBank and
    # branch") and the whole capture then failed validation.
    r"\s*[::\-]?[ \t]*([A-Z]{2}\d{2}(?:[ \t]?[A-Z0-9]{2,30}){1,8})",
    re.IGNORECASE,
)

# Anything that looks like a second labelled value on the same line. A licence
# header frequently reads "Company: X Date: 12 March 2026", and without this the
# company value would arrive carrying the date along with it.
#
# This is a closed list on purpose. A shape-based rule ("any capitalised run
# followed by a colon") also matches the middle of a company name — it read
# "Apex Gulf Technical Solutions LLC Date:" as three capitalised words before a
# colon and truncated the name to "Apex Gulf Technical". Only labels that really
# do trail a value are listed here.
_NOISE_LABELS = (
    "date", "dated", "email", "e-mail", "tel", "telephone", "phone", "mobile",
    "fax", "website", "web", "p.o. box", "po box", "ref", "reference", "no",
    "no.", "issued", "issued on", "updated", "print date", "contact", "issuer",
)
_TRAILING_LABEL_RE = re.compile(
    r"\s+(?:" + "|".join(re.escape(n) for n in _NOISE_LABELS) + r"):\s",
    re.IGNORECASE,
)

# A value that is only a document-type word means the label matched the document's
# own title rather than its content: "Company Profile" on a company profile is a
# heading, not a legal name.
_TITLE_WORDS = frozenset(
    {
        "profile", "licence", "license", "certificate", "report", "details",
        "information", "form", "schedule", "attachment", "annex", "letter",
        "confirmation", "registration", "extract", "extracts", "statement",
        "declaration", "notice", "application", "guidelines", "summary",
        "cover", "sheet", "particulars", "records", "register", "clearance",
        "acknowledgement", "acknowledgment", "undertaking", "memorandum",
    }
)

# A value ending in one of these is an entity name whatever it starts with.
# UAE free-zone forms are included alongside the usual ones because that is
# what a Dubai licence actually shows.
_LEGAL_FORM_SUFFIXES = frozenset(
    {
        "llc", "l.l.c", "llp", "lp", "ltd", "limited", "plc", "inc",
        "incorporated", "corp", "corporation", "co", "company", "gmbh", "ag",
        "sa", "s.a", "s.a.r.l", "sarl", "bv", "nv", "ab", "as", "oy", "asa",
        "pjsc", "psc", "fzc", "fze", "fzc-pjsc", "fz-llc", "fzllc", "wll",
        "est", "establishment", "pte", "pty", "sdn", "bhd", "kk", "gk",
        "kgaa", "trust", "cooperative",
    }
)


_MONTHS = {
    m: i
    for i, ms in enumerate(
        [
            ("january", "jan"), ("february", "feb"), ("march", "mar"),
            ("april", "apr"), ("may",), ("june", "jun"), ("july", "jul"),
            ("august", "aug"), ("september", "sep", "sept"),
            ("october", "oct"), ("november", "nov"), ("december", "dec"),
        ],
        start=1,
    )
    for m in ms
}

_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b")
_WORDS_RE = re.compile(
    r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b"
)
_DAY_MONTH_YEAR_RE = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"
)

# A person's name: two to four capitalised words. "Rashid Al-Farsi" and
# "Ahmed Mohammed Al Hosani" both match; a sentence does not.
_PERSON_RE = re.compile(r"^[A-Z][a-z]+(?:[- ][A-Z][a-zA-Z'’.-]+){1,3}$")
_DESIGNATION_RE = re.compile(
    r"^\s*(authoris|authoriz|signatory|signatory|managing director|"
    r"general manager|manager|director|chief executive|ceo|cfo|coo|"
    r"managing partner|partner|owner|proprietor|secretary|"
    r"authorised signatory|authorized signatory)\b",
    re.IGNORECASE,
)

# Country names worth inferring from an address. Deliberately a closed list: a
# country of incorporation is a compliance field, and it is better left blank
# than guessed at from a free-text address.
_COUNTRIES = (
    ("united arab emirates", "United Arab Emirates"),
    ("uae", "United Arab Emirates"),
    ("u.a.e.", "United Arab Emirates"),
    ("saudi arabia", "Saudi Arabia"),
    ("kingdom of saudi arabia", "Saudi Arabia"),
    ("ksa", "Saudi Arabia"),
    ("united kingdom", "United Kingdom"),
    ("uk", "United Kingdom"),
    ("india", "India"),
    ("qatar", "Qatar"),
    ("bahrain", "Bahrain"),
    ("oman", "Oman"),
    ("kuwait", "Kuwait"),
    ("singapore", "Singapore"),
    ("hong kong", "Hong Kong"),
    ("germany", "Germany"),
    ("france", "France"),
    ("netherlands", "Netherlands"),
    ("switzerland", "Switzerland"),
    ("ireland", "Ireland"),
    ("united states", "United States"),
    ("usa", "United States"),
    ("japan", "Japan"),
    ("china", "China"),
)

# TRN: 15 digits in the UAE. Used only to reject an obviously wrong capture.
_TRN_RE = re.compile(r"\b\d{15}\b")
# IBAN: 2 letters, 2 digits, then up to 30 alphanumerics, spaces allowed.
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[\s]?[A-Z0-9]{2,30}){1,8}\b")


def _label_of(field: str) -> str:
    return FIELD_LABELS.get(field, field)


def _step_of(field: str) -> int:
    return FIELD_STEP.get(field, 0)


def normalise_date(raw: str) -> Optional[str]:
    """Turn a human date into `YYYY-MM-DD`, or return None.

    The wizard's date inputs are `type="date"`, which silently displays nothing
    for a value it cannot parse. So a date is only returned when it is genuinely
    a calendar date, and an ambiguous numeric form is read day-first, which is
    what UAE paperwork uses.
    """
    text = (raw or "").strip().rstrip(".,;")
    if not text:
        return None

    def valid(year: int, month: int, day: int) -> Optional[str]:
        if not (1900 <= year <= 2100):
            return None
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        # Reject 31 April and friends, which the range check above would allow.
        import calendar

        if day > calendar.monthrange(year, month)[1]:
            return None
        return f"{year:04d}-{month:02d}-{day:02d}"

    m = _ISO_RE.search(text)
    if m:
        return valid(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = _WORDS_RE.search(text)
    if m and m.group(2).lower() in _MONTHS:
        return valid(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))

    m = _DAY_MONTH_YEAR_RE.search(text)
    if m and m.group(1).lower() in _MONTHS:
        return valid(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))

    m = _NUMERIC_RE.search(text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return valid(year, month, day)

    return None


def normalise_iban(raw: str) -> Optional[str]:
    """Strip the grouping spaces an IBAN is printed with.

    `AE07 0331 2345 6789 0123 456` and `AE070331234567890123456` are the same
    account, and the unspaced form is the one that compares equal downstream.
    """
    text = re.sub(r"\s+", "", raw or "")
    text = text.strip().strip(".,;")
    if len(text) < 15 or not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]+$", text):
        return None
    return text


def _clean_value(raw: str) -> str:
    """Tidy a captured value without changing what it says."""
    value = re.sub(r"\s+", " ", raw or "").strip()
    # Drop a trailing separator left by a label-style line.
    value = value.rstrip(" .:;-—–")
    return value


def _trim_trailing_labels(value: str) -> str:
    """Cut a value at the next `Something:` on the same line.

    "Company: Apex Gulf LLC Date: 28 September 2026" must not yield a company
    name with the issue date stuck to the end of it.
    """
    m = _TRAILING_LABEL_RE.search(value)
    if m and m.start() > 0:
        return value[: m.start()].strip()
    return value


def _plausible(field: str, value: str) -> bool:
    """Reject a capture that matched the document's structure, not its content.

    The problem this solves: a document's own heading satisfies the label it
    begins with. "Company Profile" matches the `company` label and fills the
    mandatory legal-name field with the word "Profile". So does every variant:

        Company Profile                     -> "Profile"
        Company Profile — Extract (partial)  -> "Profile — Extract (partial)"
        Company Details                     -> "Details"
        Company Information Sheet           -> "Information Sheet"

    Two-part rule, because either signal alone is wrong:

      - A trailing legal-form suffix (LLC, Ltd, FZE, GmbH, PJSC …) means the
        value really is an entity name. This has to be checked *first*: there
        are perfectly real companies called "Information Resources Trading
        LLC", and keying on the first word alone throws those away.
      - Otherwise, a first word that names a document ("Profile", "Details",
        "Certificate") means this is a heading. No registered entity begins its
        name with one.

    Leaving the field blank is the right failure. An empty mandatory field asks
    the officer a question; a wrong one does not.
    """
    if not value:
        return False
    if field != "legal_name":
        return True

    cleaned = value.strip().rstrip(".,;:").lower()
    last_word = cleaned.split()[-1] if cleaned.split() else ""
    if last_word.strip(".,;:()") in _LEGAL_FORM_SUFFIXES:
        return True

    first_word = cleaned.lstrip("—–-·•").split(" ", 1)[0]
    if first_word.strip(".,:;()[]'\"") in _TITLE_WORDS:
        return False
    return True


def _iter_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _match_label(line: str) -> List[Tuple[str, str]]:
    """Return [(field, value)] for every catalogue label starting this line.

    A single line can carry more than one labelled value, so this returns a list
    rather than bailing at the first match.

    Two details that only show up against real paperwork:

    - Labels are tried longest first. `year_of_commencement` carries both
      "year of commencement" and "year of commencement of business"; on the line
      "Year of commencement of business 2015" the short one matches first in
      source order and leaves the value as "of business 2015". Sorting makes the
      specific label win regardless of how the tuple happens to be written.

    - A label may be preceded by a qualifier. Documents say "Primary compliance
      contact" and "Bank account name", not the bare "compliance contact" and
      "account name" the tuples are keyed on. Without this, `^label` never
      matches and the field is silently missed. The qualifier set is closed and
      small on purpose — allowing arbitrary leading words would let a sentence
      like "as reported by the company address" register as an address.
    """
    found: List[Tuple[str, str]] = []
    for field, patterns in FIELD_PATTERNS.items():
        for label in sorted(patterns, key=len, reverse=True):
            pattern = re.compile(
                r"^(?:" + _LABEL_QUALIFIER + r")?" + re.escape(label)
                + r"\s*[::\-–—]?\s+(.+)$",
                re.IGNORECASE,
            )
            m = pattern.match(line)
            if m:
                value = _clean_value(_trim_trailing_labels(m.group(1)))
                if value:
                    found.append((field, value))
                break
    return found


def _infer_country(text: str) -> Optional[str]:
    lowered = f" {text.lower()} "
    for token, name in _COUNTRIES:
        if f" {token} " in lowered or f" {token}," in lowered or f", {token} " in lowered:
            return name
    return None


def extract_company_fields(text: str) -> List[ExtractedField]:
    """Read company/registration/bank details out of a document's text.

    Returns the fields it could place, best-effort and in catalogue order. A
    field that is not confidently present is simply absent from the result; the
    officer fills those by hand.
    """
    lines = _iter_lines(text)
    if not lines:
        return []

    found: Dict[str, ExtractedField] = {}

    def offer(field: str, value: str, source: str, confidence: str) -> None:
        """Record a value unless the field is already set.

        First match wins. A later, weaker occurrence — a different company named
        in a footer, a "billing address" on a later page — must not overwrite the
        value the document led with.
        """
        if not value or field in found:
            return
        if not _plausible(field, value):
            return
        found[field] = ExtractedField(
            field=field,
            value=value,
            confidence=confidence,
            label=_label_of(field),
            source=source[:220],
            step=_step_of(field),
        )

    # ---- labelled values -------------------------------------------------
    for line in lines:
        for field, value in _match_label(line):
            if field == "vat_registration_status":
                continue
            if field in ("trade_license_expiry", "date_of_incorporation"):
                iso = normalise_date(value)
                if iso:
                    offer(field, iso, line, EXACT)
                continue
            if field == "vat_registration_no":
                # Prefer a clean 15-digit TRN from anywhere in the captured value.
                digits = re.sub(r"\D", "", value)
                m = _TRN_RE.search(digits)
                offer(field, m.group(0) if m else value, line, EXACT)
                continue
            offer(field, value, line, EXACT)

    # ---- IBAN / TRN by value pattern ------------------------------------
    # Done separately from the label table because their labels print without a
    # colon and their values have a shape strict enough to recognise on sight.
    iban = normalise_iban(_IBAN_LABEL_RE.search(text).group(1)) if _IBAN_LABEL_RE.search(text) else None
    if iban:
        m = _IBAN_LABEL_RE.search(text)
        line = next(
            (ln for ln in lines if "iban" in ln.lower()),
            m.group(0),
        )
        offer("bank_iban", iban, line, EXACT)
    if "vat_registration_no" not in found:
        m = _TRN_LABEL_RE.search(text)
        if m:
            digits = re.sub(r"\D", "", m.group(1))
            t = _TRN_RE.search(digits)
            if t:
                line = next(
                    (ln for ln in lines if "trn" in ln.lower()
                     or "tax registration" in ln.lower()
                     or "vat registration" in ln.lower()),
                    m.group(0),
                )
                offer("vat_registration_no", t.group(0), line, EXACT)

    # ---- VAT status ------------------------------------------------------
    # Set from an explicit status phrase, else from the presence of a TRN. The
    # client needs this to be "registered" or the TRN field stays hidden behind
    # its conditional, which would look like the extraction silently failed.
    if "vat_registration_status" not in found:
        lowered = text.lower()
        for line in lines:
            low = line.lower()
            if "not registered for vat" in low or "vat exempt" in low:
                offer(
                    "vat_registration_status",
                    "not_registered" if "not registered" in low else "exempt",
                    line,
                    INFERRED,
                )
                break
            if "registered for vat" in low or re.search(r"\bvat\s+registered\b", low):
                offer("vat_registration_status", "registered", line, INFERRED)
                break
        else:
            if "vat_registration_no" in found:
                offer(
                    "vat_registration_status",
                    "registered",
                    found["vat_registration_no"].source,
                    INFERRED,
                )

    # ---- signatory --------------------------------------------------------
    # Licence and profile documents put the name on one line and the role on the
    # next: "Rashid Al-Farsi" / "Authorised signatory".
    for i, line in enumerate(lines[:-1]):
        nxt = lines[i + 1]
        if not _PERSON_RE.match(line):
            continue
        if not _DESIGNATION_RE.match(nxt):
            continue
        offer("authorized_representative_name", _clean_value(line), line, INFERRED)
        offer(
            "authorized_representative_designation",
            _clean_value(nxt),
            nxt,
            INFERRED,
        )
        break

    # ---- country ----------------------------------------------------------
    if "country_of_incorporation" not in found:
        country = _infer_country(text)
        if country:
            anchor = found.get("registered_address")
            offer(
                "country_of_incorporation",
                country,
                anchor.source if anchor else "Country named in the document",
                INFERRED,
            )

    # Catalogue order, so the review list is stable between documents.
    return [
        found[f] for f in FIELD_LABELS if f in found
    ]
