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
from dataclasses import dataclass, replace
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
    # 2. Nature of business & product range
    "nature_of_business": "Nature of business",
    "goods_services_proposed": "Goods / services proposed",
    "geographic_coverage": "Geographic coverage",
    "supply_type": "Supply type",
    "sensitivity": "Nature sensitivity",
    # 3. Authorized representative (extra detail beyond name/designation)
    "authorized_representative_id": "Representative ID / Emirates ID",
    "authorized_representative_id_expiry": "Representative ID expiry",
    "authorized_representative_authority_basis": "Authority basis",
    # 4. Directors / owners (scalar parts)
    "ultimate_parent_company": "Ultimate parent company",
    "ownership_structure": "Ownership structure",
    "bearer_shares": "Bearer shares",
    "nominee_shareholders": "Nominee shareholders",
    "pep_present": "PEP present",
    "pep_family_member": "PEP family member",
    "pep_close_associate": "PEP close associate",
    "ownership_structure_chart": "Ownership structure chart reference",
    # 5. References (scalar parts)
    "public_review_links": "Public review links",
    # 6. Annual sales turnover
    "turnover_year_1": "Turnover year 1 (AED)",
    "turnover_year_2": "Turnover year 2 (AED)",
    "turnover_year_3": "Turnover year 3 (AED)",
    "number_of_employees": "Number of employees",
    "financial_statements_audited": "Financial statements audited",
    # 7. Bank & payment (extras)
    "swift_code": "SWIFT / BIC code",
    "payment_terms": "Payment terms",
    "payment_structure": "Payment structure",
    "third_party_payment": "Third-party payment",
    "foreign_account_payment": "Foreign account payment",
    "cash_payment": "Cash payment",
    # 8. FTA supplier risk & commercial disclosures
    "engages_government_officials": "Engages government officials",
    "government_licensing": "Government licensing required",
    # 12. Supplier declaration & signature
    "supplier_declaration": "Supplier declaration",
    "authorized_signatory_name": "Authorized signatory name",
    "authorized_signatory_designation": "Authorized signatory designation",
    "date_signed": "Date signed",
}

# Sections whose target form field is a repeatable list of rows, not a single
# value. The flat catalogue above cannot hold them (one value per key), so the
# parser reads them separately into `sections` and the client appends them into
# the matching row arrays. Each spec maps a row label to the sub-fields the
# wizard's row type expects, in the order the document prints them.
LIST_SECTIONS: Dict[str, Dict[str, Any]] = {
    "directors_and_owners": {
        "label": "Directors, partners, shareholders & owners",
        "step": 2,
        # "Director: Name — Position — Nationality" or a bare name line under a
        # "Directors" heading. Sub-fields map onto DirectorRow.
        "row_label": "director",
        "fields": ("name", "position", "nationality"),
    },
    "ubos": {
        "label": "Ultimate beneficial owners",
        "step": 2,
        "row_label": "ubo",
        "fields": ("name", "nationality", "ownership_percentage"),
    },
    "client_references": {
        "label": "Major client references",
        "step": 2,
        "row_label": "client",
        "fields": ("client_name", "contact_details"),
    },
    "products_services": {
        "label": "Products & services listing",
        "step": 1,
        "row_label": "product",
        "fields": ("name",),
    },
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
        "legal & trading name", "legal and trading name",
        "legal name & trading name", "legal trading name",
        "company / legal name", "legal name / company name",
        "legal name", "legal entity name", "registered name",
        "name of company", "company name", "company", "licensee", "licence holder",
        "license holder", "taxable person",
    ),
    "trade_license_no": (
        "reg & trade licence", "reg & trade license",
        "registration & trade licence", "registration & trade license",
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
        "jurisdiction & type", "jurisdiction and type",
        "country of incorporation", "country of registration",
        "jurisdiction", "country", "jurisdiction of incorporation",
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
        "primary contact person", "primary contact", "main contact",
        "contact person", "point of contact",
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
        "bank & terms", "bank and terms",
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
    # 2. Nature of business & product range
    "nature_of_business": (
        "category & goods", "category and goods", "category & goods/services",
        "goods & category", "category and goods offered",
        "nature of business", "business nature", "line of business",
        "business activity", "business description", "activity",
    ),
    "goods_services_proposed": (
        "goods / services proposed", "goods and services proposed",
        "goods services proposed", "products and services proposed",
        "proposed goods and services", "scope of supply", "goods and services",
        "description of goods and services", "supplied products",
    ),
    "geographic_coverage": (
        "geographic coverage", "coverage", "geographical coverage",
        "markets served", "regions served",
    ),
    "supply_type": (
        "supply type", "type of supply", "supply category",
    ),
    "sensitivity": (
        "category sensitivity", "nature sensitivity", "sensitivity",
        "sensitivity level",
    ),
    # 3. Representative detail
    "authorized_representative_id": (
        # Longest first, or "Emirates ID reference 784-…" matches the bare
        # "emirates id" label and files the word "reference" as the ID number.
        "representative emirates id reference", "emirates id reference",
        "emirates id ref", "emirates id number", "representative id",
        "emirates id", "id number", "identity number", "passport number",
    ),
    "authorized_representative_id_expiry": (
        "representative id expiry", "emirates id expiry", "id expiry",
        "identity expiry", "passport expiry",
    ),
    "authorized_representative_authority_basis": (
        "authority basis", "basis of authority", "authority",
        "power of attorney", "board resolution",
    ),
    # 4. Ownership scalars
    "ultimate_parent_company": (
        "ultimate parent company", "parent company", "ultimate parent",
    ),
    "ownership_structure": (
        "declared ownership structure", "ownership structure",
        "structure of ownership", "ownership type",
    ),
    "bearer_shares": (
        "bearer shares", "bearer share",
    ),
    "nominee_shareholders": (
        "nominee shareholders", "nominee shareholder", "nominee",
    ),
    "pep_present": (
        # "PEP directly involved" has to be listed ahead of any shorter PEP
        # phrase: matching "pep" alone left the value as "directly involved No",
        # which the Yes/No normaliser then rejected and the field stayed blank.
        "pep directly involved", "directly involved pep",
        "pep present", "politically exposed", "pep status",
    ),
    "pep_family_member": (
        "pep family involved", "family involved pep",
        "pep family member", "pep family", "family member pep",
    ),
    "pep_close_associate": (
        "close associate pep", "pep close associate", "close associate",
        "pep associate",
    ),
    "ownership_structure_chart": (
        "ownership chart reference", "ownership chart ref", "ownership chart",
        "ownership structure chart", "organisation chart", "organization chart",
        "org chart", "structure chart", "ownership diagram",
    ),
    # 5. References scalar
    "public_review_links": (
        "public review links", "review links", "public links",
    ),
    # 6. Turnover
    "turnover_year_1": (
        "turnover year 1", "annual turnover year 1", "turnover 1",
        "year 1 turnover", "sales turnover year 1",
    ),
    "turnover_year_2": (
        "turnover year 2", "annual turnover year 2", "turnover 2",
        "year 2 turnover", "sales turnover year 2",
    ),
    "turnover_year_3": (
        "turnover year 3", "annual turnover year 3", "turnover 3",
        "year 3 turnover", "sales turnover year 3",
    ),
    "number_of_employees": (
        "number of employees", "employees", "staff strength", "headcount",
    ),
    "financial_statements_audited": (
        "financial statements audited", "audited financial statements",
        "financial statements", "audited",
    ),
    # 7. Payment scalars
    "swift_code": (
        "swift", "swift code", "swift / bic", "bic", "bic code",
        "swift bic code",
    ),
    "payment_terms": (
        "payment terms", "terms of payment",
    ),
    "payment_structure": (
        "payment structure", "type of payment", "payment type",
    ),
    "third_party_payment": (
        "third-party payment", "third party payment",
    ),
    "foreign_account_payment": (
        "foreign account payment", "foreign payment",
    ),
    "cash_payment": (
        "cash payment",
    ),
    # 8. FTA supplier risk & commercial disclosures. These print as questions
    # ("Does the supplier interact with government officials? Yes"), so the
    # labels name the subject rather than the field, and `_canon` turns the
    # answer into the literal the wizard's <select> stores.
    "engages_government_officials": (
        "interacts with government officials", "interacts with government",
        "interacts with govt", "interact with government officials",
        "engages government officials", "engages with government officials",
        "engagement with government officials", "government official contact",
        "deals with government officials", "government interaction",
        "public sector interaction", "government officials",
    ),
    "government_licensing": (
        "govt licence required", "govt license required",
        "government licence required", "government license required",
        "government licensing required", "licence required from government",
        "license required from government", "government licensing",
        "govt licensing", "requires government licence",
        "requires government license", "government licence",
        "government license", "govt licence", "govt license",
    ),
    # 12. Declaration & signature
    "supplier_declaration": (
        "supplier declaration", "declaration",
    ),
    "authorized_signatory_name": (
        "authorized signatory name", "authorised signatory name",
        "signatory name",
    ),
    "authorized_signatory_designation": (
        "authorized signatory designation", "authorised signatory designation",
        "signatory designation", "signatory title",
    ),
    "date_signed": (
        "date signed", "signed on", "signature date",
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

# A value that opens with one of these is a document heading continuing past its
# label ("Supplier Declaration & Signature"), not a field value. Anchored so it
# only fires at the very start of a captured value.
_HEADING_CONNECTOR = re.compile(
    r"\s*(?:&(?=\s)|\b(?:and|of|the)\b)", re.IGNORECASE
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
    # Hyphens and slashes as well as spaces: a licence prints "15-Nov-2027"
    # and the wizard's date input shows nothing at all for a value it cannot
    # parse, so a hyphenated date used to be silently dropped.
    r"\b(\d{1,2})[-\s]{1,3}([A-Za-z]{3,9})\.?[-\s]{1,3}(\d{4})\b"
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
    # Drop a trailing separator left by a label-style line. A bare "|" is a
    # sub-field the primary-value split left behind ("Tariq Mansoor |"), never
    # something an officer would have typed.
    value = value.rstrip(" .:;-—–|")
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


# Ligatures and typographic quotes that a PDF text layer leaves behind. A
# licence printed with an "fi" ligature yields one U+FB01 glyph, so a label
# search for "configured" or "qualified" silently missed every occurrence.
_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\ufb01": "fi",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ",
})

# A line that opens a new logical entry. Anything else is read as the wrapped
# tail of the entry above it.
#
# The problem this solves: a data sheet breaks a long value across lines, and
# the label that owns it is on the previous line —
#
#     Major References 1. ADNOC … | 2. TAQA Group (AED 3.5M) | 3. Aldar (AED
#     2M)
#     Bank & Terms First Abu Dhabi Bank (FAB) | IBAN: AE39… | Account Name: Apex
#     Industrial Solutions LLC | Net 30 Days
#
# Reading the file line by line threw "2M)" and "Industrial Solutions LLC"
# away, and with them the bank name, the account holder and the third client.
_LINE_STARTERS = (
    re.compile(r"^scenario\s*#", re.IGNORECASE),
    re.compile(r"^step\s*\d", re.IGNORECASE),
    # An ALL-CAPS heading: PURPOSE, TOTAL SCENARIOS, STEP 3: OWNERSHIP.
    re.compile(r"^[A-Z][A-Z0-9 &/.,'()\d\-]{2,80}$"),
    # A label printed with a colon: "Expected AI Output: …".
    re.compile(r"^[A-Za-z][\w&/.,'() \-]{0,60}:\s"),
)

# Label phrases that open a logical line, longest first so the specific one
# wins. Built from the field catalogue and the row-group headings, then extended
# with the compound labels a data sheet prints, so adding a label to the
# catalogue is enough to have it reflow correctly.
_LABEL_STARTERS: List[str] = sorted(
    {
        pattern.lower()
        for patterns in FIELD_PATTERNS.values()
        for pattern in patterns
    }
    | {spec["label"].lower() for spec in LIST_SECTIONS.values()}
    | {
        "legal & trading name", "legal and trading name",
        "reg & trade licence", "reg & trade license",
        "jurisdiction & type", "jurisdiction and type",
        "bank & terms", "bank and terms",
        "3-year turnover", "3 year turnover", "3-year annual turnover",
        "owners / directors", "owners/directors", "owners and directors",
        "major references", "major client references", "major clients",
        "category & goods", "category and goods",
        "primary contact", "contact person", "main contact",
        "expected ai output",
    },
    key=len,
    reverse=True,
)


def _starts_logical_line(line: str) -> bool:
    if any(pattern.match(line) for pattern in _LINE_STARTERS):
        return True
    lowered = line.lower()
    for label in _LABEL_STARTERS:
        if not lowered.startswith(label):
            continue
        rest = lowered[len(label):]
        # The label has to end on a word boundary, or "companies" would open a
        # line on the "company" label.
        if not rest or not rest[0].isalpha():
            return True
    return False


def _is_incomplete(line: str) -> bool:
    """True when a line stops mid-value, so the line below it is its wrap.

    A wrapped row is only recognisable by what it does *not* end with. Two real
    layouts look identical at the label — the data sheet packs sub-fields onto
    one pipe-separated row, the forms print "Licence number CN-1094821" and
    "Licence type Limited Liability Company" as separate rows — so asking
    whether a line opens a field cannot tell them apart. Asking whether the line
    finishes one can:

        … | 3. Aldar Properties (AED      -> an unclosed bracket
        … | 2. Saeed Al-Khouri (Director, UAE,   -> a trailing comma
        … | Account Name: Apex            -> a compound row cut mid-segment

    "Licence number CN-1094821" and "Bank and branch Emirates NBD, Dubai
    Marina Branch" are both finished, and are left alone.
    """
    if line.endswith(("(", "[", "{", ",", ":", "|", "-", "–", "—")):
        return True
    if line.count("(") > line.count(")") or line.count("[") > line.count("]"):
        return True
    # A compound row broken inside its last segment. A plain single-value line
    # never carries a pipe, so this cannot reach out of a compound row.
    return "|" in line and not line.endswith((")", ".", ";", "!", "?"))


def _reflow(lines: List[str]) -> List[str]:
    """Stitch a wrapped value back onto the line that owns it."""
    out: List[str] = []
    for line in lines:
        if (
            out
            and _is_incomplete(out[-1])
            and not _starts_logical_line(line)
        ):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return out


def _iter_lines(text: str) -> List[str]:
    normalised = (text or "").translate(_LIGATURES)
    lines = [ln.strip() for ln in normalised.splitlines() if ln.strip()]
    return _reflow(lines)


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
            # A separator is required only when one is printed. A label that
            # runs straight into its value ("Licensee Apex Gulf LLC") keeps
            # working off the \s+; a label followed by a colon takes only what
            # is on its own line ("Supplier Declaration: Yes" and never the
            # heading that trails "& Signature").
            pattern = re.compile(
                r"^(?:" + _LABEL_QUALIFIER + r")?" + re.escape(label)
                + r"(?:\s*[::\-–—]\s*(.+)|\s+(.+))$",
                re.IGNORECASE,
            )
            m = pattern.match(line)
            if m:
                raw_value = m.group(1) if m.group(1) is not None else m.group(2)
                # A section heading printed as "Label & Rest" or "Label and
                # Rest" makes the label itself match with the connector as the
                # start of the value ("Supplier Declaration: & Signature"). A
                # real value never opens with one of these, so this rejects the
                # heading without touching "Goods and services proposed", where
                # the connector sits after the value has already begun.
                if _HEADING_CONNECTOR.match(raw_value):
                    continue
                value = _clean_value(
                    _primary_segment(_trim_trailing_labels(raw_value))
                )
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


#: Fields that hold a person's name. A document prints "Tariq Mansoor
#: (Procurement Director)"; the role belongs in the designation field, which is
#: read separately, and leaving it in the name puts a job title in a field the
#: form validates as a person.
_PERSON_NAME_FIELDS = frozenset({
    "authorized_representative_name", "authorized_signatory_name",
})

#: The wizard's Select3 fields store the literal strings "Yes" / "No".
_YES_NO_FIELDS = frozenset({
    "bearer_shares", "nominee_shareholders", "pep_present",
    "pep_family_member", "pep_close_associate", "financial_statements_audited",
    "supplier_declaration", "engages_government_officials",
    "government_licensing",
})

#: The wizard's <select> fields store the option *key*, never the label. A
#: document prints the human phrase ("Goods", "Simple — direct"), so the raw
#: capture has to be mapped or the select renders blank and the risk engine
#: mis-keys the factor. Anything not in the table is left as captured.
_SELECT_KEYS: Dict[str, Dict[str, str]] = {
    "supply_type": {
        "goods": "goods", "goods supply": "goods",
        "services": "services", "service": "services",
        "goods and services": "goods and services",
        "goods & services": "goods and services",
    },
    "sensitivity": {
        "standard": "standard", "standard / low sensitivity": "standard",
        "low": "standard", "low sensitivity": "standard",
        "moderate": "moderate", "moderate sensitivity": "moderate",
        "sensitive": "sensitive", "sensitivity": "sensitive",
        "highly sensitive": "highly_sensitive",
        "critical": "critical", "critical / prohibited category": "critical",
    },
    "ownership_structure": {
        "simple": "simple", "simple — direct, single holder": "simple",
        "direct": "simple", "single holder": "simple",
        "domestic layered": "domestic_layered",
        "domestic-layered": "domestic_layered",
        "cross border layered": "cross_border_layered",
        "cross-border layered": "cross_border_layered",
        "layered": "layered", "nominee": "nominee",
        "bearer share": "bearer_share", "bearer shares": "bearer_share",
        "offshore": "offshore", "trust": "trust",
    },
    "payment_structure": {
        "standard": "standard", "advance": "advance",
        "milestone": "milestone", "retainer": "retainer",
        "time and materials": "time_and_materials",
        "time & materials": "time_and_materials",
        "fixed price": "fixed_price", "fixed": "fixed_price",
    },
}

#: Every field that goes through `_canon`, so the loop only touches fields the
#: document could have produced.
_CANONICAL_FIELDS = (
    _YES_NO_FIELDS
    | frozenset(_SELECT_KEYS)
    | frozenset({
        "turnover_year_1", "turnover_year_2", "turnover_year_3",
        "number_of_employees",
    })
)


def _parse_row(line: str, spec: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Parse one repeatable row from a single line.

    The document prints a row either as a labelled group on one line —
    "Director: Rashid Al-Farsi — Managing Director — United Arab Emirates" — or
    as a bare value line directly under the section heading. Fields are split
    on em/en dashes and read positionally into the row's sub-fields; a trailing
    ownership percentage is peeled off UBO rows however it is written.
    """
    fields: Tuple[str, ...] = tuple(spec["fields"])
    # A row is only a row when it carries its group's prefix. Requiring the
    # prefix is what keeps a scalar that happens to sit under the heading
    # ("Ultimate Parent Company: N/A") from being read as a director. The
    # singular label is accepted alongside the plural a document usually prints.
    prefix = re.compile(
        r"^\s*" + re.escape(spec["row_label"])
        + r"s?\s*[::\-–—]\s*(?=\S)",
        re.IGNORECASE,
    )
    m = prefix.match(line)
    if not m:
        return None
    value = line[m.end():]
    # Peel a trailing ownership percentage off a UBO row before the positional
    # split, so it is never read as one of the positional fields. Documents
    # print it at the end ("… — 60%") or straight after the name
    # ("… — 60% — United Kingdom"); the field list is positional, so without
    # this the "60%" lands in `nationality` and `ownership_percentage` is empty.
    pct = ""
    if "ownership_percentage" in fields:
        pm = re.search(r"(\d{1,3})\s*%", value)
        if pm:
            pct = pm.group(1)
            value = (value[: pm.start()] + value[pm.end():]).strip(" -–—:,")

    parts = [p.strip() for p in re.split(r"\s*[–—]\s*|\s+-\s+", value) if p.strip()]
    row: Dict[str, str] = {}
    if len(parts) == 1 and len(fields) >= 1:
        row[fields[0]] = parts[0]
    else:
        for key, part in zip(fields, parts):
            if key == "ownership_percentage" and not pct:
                continue
            row[key] = part
    if pct and "ownership_percentage" in fields and "ownership_percentage" not in row:
        row["ownership_percentage"] = pct
    # A row needs at least a name to be real; drop empty shells.
    if not row.get(fields[0]):
        return None
    # Fill any missing sub-fields with empty strings so the row shape is stable.
    for key in fields:
        row.setdefault(key, "")
    return row


def _extract_list_sections(lines: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Read the repeatable row groups out of the document's lines.

    Walks the lines once and files each row by its prefix, so a group switch
    needs no heading: "UBO:" rows follow "Director:" rows directly. A row's
    prefix decides both the target group and the field layout, which keeps a
    scalar sitting between two row groups from being read as a row. Headings
    still seed the group, so a section whose rows carry no prefix is collected
    until the next heading.
    """
    out: Dict[str, List[Dict[str, str]]] = {k: [] for k in LIST_SECTIONS}

    # Prefix -> (section key, spec): the source of truth for which group a
    # prefixed row line belongs to.
    by_prefix: Dict[str, Tuple[str, Dict[str, Any]]] = {
        spec["row_label"].lower(): (key, spec)
        for key, spec in LIST_SECTIONS.items()
    }

    # Lowercased heading phrase -> section key. Built from the spec labels, then
    # extended with the plain singular/plural forms a document also uses.
    headings: Dict[str, str] = {
        spec["label"].lower(): key for key, spec in LIST_SECTIONS.items()
    }
    headings.update({
        "directors, partners, shareholders & owners": "directors_and_owners",
        "directors, partners, shareholders and owners": "directors_and_owners",
        "directors and owners": "directors_and_owners",
        "directors": "directors_and_owners",
        "owners": "directors_and_owners",
        "shareholders": "directors_and_owners",
        "ultimate beneficial owners": "ubos",
        "beneficial owners": "ubos",
        "major client references": "client_references",
        "client references": "client_references",
        "clients": "client_references",
        "references": "client_references",
        "products & services listing": "products_services",
        "products and services listing": "products_services",
        "products & services": "products_services",
        "products and services": "products_services",
        "products": "products_services",
    })

    current: Optional[str] = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        low = stripped.lower().rstrip(" :")
        if low in headings:
            current = headings[low]
            continue

        # Route by the row's own prefix, not by the open heading: this is what
        # lets one group's rows follow another's with no heading between.
        prefix_key = low.split(":", 1)[0].strip() if ":" in low else ""
        target, spec = by_prefix.get(prefix_key, (current, None))
        if target is None or spec is None:
            continue

        # Only a prefixed row line belongs to the current group. Everything
        # else — scalars, sub-headings, prose — is left to the scalar pass.
        row = _parse_row(stripped, spec)
        if row is not None:
            out[target].append(row)
    return {k: v for k, v in out.items() if v}


def _split_segments(value: str) -> List[str]:
    return [seg.strip() for seg in (value or "").split("|") if seg.strip()]


# A segment that carries a recognised sub-label rather than a bare value. Used
# to tell a compound line ("Bank & Terms First Abu Dhabi Bank (FAB) | IBAN: … |
# Account Name: …") from a plain value that merely happens to contain a pipe.
_COMPOUND_SUB_LABEL = re.compile(
    r"^(?:expiry|expiration|expires|valid\s+(?:until|till)|trn|vat\s+registration"
    r"|iban|account\s+(?:name|number|no|holder)|beneficiary\s+name|swift|bic"
    r"|e?-?mail|phone|tel(?:ephone)?|mobile|fax|payment\s+terms|terms"
    r"|primary\s+category|category|issue\s+date|issued(?:\s+on)?|no\.?|number"
    r"|share|ownership|role|position|country|jurisdiction"
    # A legal form prints beside the jurisdiction and is not part of it:
    # "United Arab Emirates (Abu Dhabi mainland) | Limited Liability Company".
    r"|limited|llc|l\.l\.c|ltd|inc|incorporated|corp(?:oration)?|co(?:mpany)?"
    r"|fze|fzc|pjsc|psc|est(?:ablishment)?|pte|pty|sdn|bhd|gmbh|ag"
    r"|sole\s+proprietor(?:ship)?|civil\s+company|partnership|branch"
    r"|free\s+zone|mainland|holding(?:\s+company)?"
    r")\b",
    re.IGNORECASE,
)


def _primary_segment(value: str) -> str:
    """The labelled value itself, without the sub-fields packed beside it.

    A data sheet prints one row per field, so a compound row's first segment is
    the field's own value and everything after it is a sub-field. Only split
    when a later segment really does carry a sub-label — a value that merely
    contains a pipe ("A | B") is left whole.
    """
    segments = _split_segments(value)
    if len(segments) < 2:
        return value
    if any(_COMPOUND_SUB_LABEL.match(seg) for seg in segments[1:]):
        return segments[0]
    return value


#: Sub-fields that a compound line packs beside its own value, keyed by the
#: field they belong to. Run against every logical line, independently of the
#: label table: the label tells the parser which row this is, but it cannot say
#: how many fields the row carries.
_COMPOUND_FIELDS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("trade_license_expiry", re.compile(
        # An identity document carries an expiry too, and its line already
        # belongs to authorized_representative_id_expiry. Without these
        # guards "Representative ID Expiry: 2029-08-30" is filed as the trade
        # licence date whenever the licence line is absent — which is exactly
        # the shape an incomplete submission has, so the gap it should have
        # reported gets quietly filled with the wrong date. A licence row
        # ("Trade Licence Expiry: …", or "… | Expiry: …" beside its number) is
        # untouched: none of these words precede its expiry.
        r"(?<!representative id )(?<!emirates id )(?<!passport )"
        r"(?<!identity )(?<!id )"
        r"\b(?:expiry|expiration|expires|valid\s+until|valid\s+till)"
        r"\s*[::\-]?\s*"
        r"(\d{1,2}[-/\s][A-Za-z]{3,9}[-/\s]\d{4}|\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
        re.IGNORECASE,
    )),
    ("vat_registration_no", re.compile(
        r"\b(?:TRN|Tax\s+Registration\s+Number|VAT\s+Registration)"
        r"\s*[::\-]?\s*(\d[\d \t]{10,24})",
        re.IGNORECASE,
    )),
    ("bank_iban", re.compile(
        r"\bIBAN\s*[::\-]?\s*([A-Z]{2}\d{2}(?:[ \t]?[A-Z0-9]{2,30}){1,8})",
        re.IGNORECASE,
    )),
    ("bank_account_name", re.compile(
        r"\b(?:account\s+name|account\s+holder|beneficiary\s+name)"
        r"\s*[::\-]\s*(.+?)(?:\s*\||\s{2,}|$)",
        re.IGNORECASE,
    )),
    ("bank_account_number", re.compile(
        r"\b(?:account\s+number|account\s+no\.?)\s*[::\-]\s*([\d \-]{6,24})",
        re.IGNORECASE,
    )),
    ("swift_code", re.compile(
        r"\b(?:SWIFT|BIC)(?:\s+Code)?\s*[::\-]?\s*([A-Z0-9]{8,11})\b",
        re.IGNORECASE,
    )),
    ("compliance_contact", re.compile(
        r"\b(?:e-?mail)\s*[::\-]\s*([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})",
        re.IGNORECASE,
    )),
    ("payment_terms", re.compile(
        r"\b(?:payment\s+terms|terms)\s*[::\-]\s*(.+?)(?:\s*\||$)",
        re.IGNORECASE,
    )),
    ("payment_terms", re.compile(
        # "… | Net 30 Days" prints the terms with no label at all.
        r"\bNet\s+\d+\s*(?:Days?|Months?|Weeks?)?\b",
        re.IGNORECASE,
    )),
    ("authorized_representative_designation", re.compile(
        # "Tariq Mansoor (Procurement Director) | email: …" — the role sits in
        # the bracket straight after the name, and nowhere has it a label.
        #
        # The bracket has to contain a role word. Without that requirement the
        # document's own header qualified: "10 Test Suites (Bad to Good)" filled
        # the designation with "Bad to Good". The whole bracket is captured, so
        # "Procurement Director" survives intact.
        r"^[A-Z][\w'’.\- ]{2,60}\s*\((?=.{0,40}?\b(?:"
        r"director|manager|chief|ceo|cfo|coo|cto|cmo|partner|owner"
        r"|proprietor|secretary|procurement|purchasing|signatory"
        r"|executive|officer|head|lead|finance|accounts|commercial"
        r")\b)([^)]{2,60})\)\s*(?:\||$)",
        re.IGNORECASE,
    )),
)

#: Turnover printed as a dated series on one row: "3-Year Turnover 2023: AED
#: 18,500,000 | 2024: AED 22,000,000 | 2025: AED 26,400,000". There is no label
#: for any single year, so the years have to be paired with their amounts and
#: ordered before they can be filed.
_TURNOVER_SERIES_RE = re.compile(
    r"\b(19|20)\d{2}\s*[::\-]\s*"
    r"(?:AED|USD|EUR|GBP|SAR|INR|AED\s*)?\s*"
    r"([\d][\d,]{2,}(?:\.\d+)?)",
    re.IGNORECASE,
)


def _compound_values(line: str) -> List[Tuple[str, str]]:
    """Every sub-field this line carries, as (field, value) pairs."""
    out: List[Tuple[str, str]] = []
    for field, pattern in _COMPOUND_FIELDS:
        m = pattern.search(line)
        if m:
            value = _clean_value(m.group(1 if m.groups() else 0))
            if value:
                out.append((field, value))
    return out


def _turnover_series(line: str) -> List[Tuple[str, str]]:
    """Pair a dated turnover series with the wizard's year slots.

    The wizard stores "Year 1 / 2 / 3", which read earliest-first on the real
    form, so the series is sorted by year before it is filed. Two entries at
    the same year keep the first.
    """
    pairs: Dict[str, str] = {}
    for m in _TURNOVER_SERIES_RE.finditer(line):
        year, amount = m.group(0), m.group(2)
        pairs.setdefault(year, re.sub(r"[^\d]", "", amount))
    if len(pairs) < 2:
        return []
    ordered = [pairs[y] for y in sorted(pairs)]
    slots = ("turnover_year_1", "turnover_year_2", "turnover_year_3")
    return list(zip(slots, ordered[: len(slots)]))


def _numbered_items(value: str) -> List[str]:
    """The "1. Name (Role, Country, 60% Share) | 2. …" items on one row."""
    items = _split_segments(value)
    if len(items) < 2:
        return []
    out: List[str] = []
    for item in items:
        m = re.match(r"^\s*\d+\s*[.)]\s*(.+)$", item)
        out.append((m or item).group(1 if m else 0).strip() if m else item.strip())
    return [i for i in out if i]


#: A single labelled row that carries a whole numbered group. A data sheet
#: prints the owners, the clients or the products on one line rather than one
#: row per line, so the row parser — which routes by a per-line prefix — never
#: sees them. The label decides the target group.
_COMPOUND_SECTIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("directors_and_owners", (
        "owners / directors", "owners/directors", "owners and directors",
        "directors / owners", "directors & owners", "shareholders / directors",
        "directors, shareholders & owners", "directors and shareholders",
        "shareholders & directors", "owners and shareholders",
    )),
    ("ubos", (
        "ultimate beneficial owners", "ultimate beneficial owner",
        "ubo", "ubos", "beneficial owners",
    )),
    ("client_references", (
        "major references", "major client references", "client references",
        "key references", "major clients", "key clients", "clients",
    )),
    ("products_services", (
        "products & services listing", "products and services listing",
        "products & services", "products and services", "product range",
        "product list", "products offered",
    )),
)

#: Words that mark a bracketed part as a nationality rather than a role.
_NATIONALITY_HINTS = re.compile(
    r"\b(?:emirati?|uae|u\.a\.e|united\s+arab\s+emirates|british|uk|indian"
    r"|pakistan(?:i)?|bangladeshi|saudi|egyptian|jordanian|lebanese"
    r"|american|german|french|dutch|irish|canadian|australian|russian"
    r"|chinese|filipino|philippine(?:s)?|kenyan|nigerian|sudanese"
    r"|stateless|state\s+less)\b",
    re.IGNORECASE,
)


def _parse_numbered_row(item: str, spec: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Read "Name (Role, Country, 60% Share)" into the group's sub-fields.

    The bracketed parts are read by what they *are*, not by position: the same
    group prints "Director, UAE, 60% Share" and "Offshore Shell, 30%", and a
    positional split would put the country in `position` on one row and the
    role in `nationality` on the next.
    """
    fields = tuple(spec["fields"])
    # Only the first bracket is the row's own. A row annotated with a note —
    # "Saeed Al-Khouri (Director, UAE, 40% Share) — No PEP hits" — carries a
    # second bracket that belongs to the note, not to the person.
    name, _, bracket = item.partition("(")
    bracket = bracket.split(")", 1)[0]
    name = _clean_value(name)
    if not name:
        return None

    row: Dict[str, str] = {key: "" for key in fields}
    row[fields[0]] = name

    pct = re.search(r"(\d{1,3})\s*%", bracket)
    if pct:
        # Stashed even when the group has no ownership column of its own, so an
        # owner listed under "Owners / Directors" can still be promoted to a
        # beneficial owner below. The extra key is dropped before the row is
        # returned to the caller.
        row["ownership_percentage"] = pct.group(1)

    for part in [p.strip() for p in bracket.split(",") if p.strip()]:
        if pct and part == pct.group(0):
            continue
        if "nationality" in fields and _NATIONALITY_HINTS.search(part):
            row["nationality"] = _clean_value(part)
        elif "position" in fields and not row.get("position"):
            # First unmatched part is the role. Later ones are notes, and
            # overwriting put "40% Share — No PEP hits" where "Director" was.
            row["position"] = _clean_value(part)
        elif "contact_details" in fields and not row.get("contact_details"):
            row["contact_details"] = _clean_value(part)
    return row


_SCENARIO_RE = re.compile(r"^\s*Scenario\s*#", re.IGNORECASE | re.MULTILINE)


def scenario_count(text: str) -> int:
    """How many vendor records a document holds. 0 for an ordinary form."""
    return len(_SCENARIO_RE.findall(text or ""))


def _first_scenario(text: str) -> str:
    """Cut a multi-record sheet down to the record it leads with.

    A data sheet prints one vendor after another ("Scenario #1: Apex …",
    "Scenario #2: Bharat …"). Reading every record into a single form builds a
    vendor that exists nowhere: Apex's licence number beside Bharat's bank
    account, and an owner list carrying all ten companies' shareholders.

    The scalar fields were already safe — first match wins — but the row
    groups accumulate, so the cut is what keeps one form one vendor.
    """
    if scenario_count(text) < 2:
        return text
    return text[: [m.start() for m in _SCENARIO_RE.finditer(text)][1]]


def _compound_sections(lines: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Collect the row groups a data sheet prints as one numbered line each."""
    out: Dict[str, List[Dict[str, str]]] = {}
    by_label: Dict[str, str] = {
        label: key for key, labels in _COMPOUND_SECTIONS for label in labels
    }
    # Longest label first, so "directors, shareholders & owners" wins over the
    # bare "directors" that prefixes it.
    ordered = sorted(by_label.items(), key=lambda kv: len(kv[0]), reverse=True)

    for line in lines:
        if "|" not in line:
            continue
        low = line.lower()
        # The label can be multi-word ("Owners / Directors 1. Tariq …"), so cut
        # it by length rather than by the first whitespace run.
        match = next(
            ((label, target) for label, target in ordered if low.startswith(label)),
            None,
        )
        if match is None:
            continue
        label, key = match
        items = _numbered_items(line[len(label):].strip())
        if not items:
            continue
        spec = LIST_SECTIONS[key]
        rows = [r for r in (_parse_numbered_row(item, spec) for item in items) if r]
        if rows:
            out.setdefault(key, []).extend(rows)

    # An owner with a stated share is also a beneficial owner. The form asks for
    # both lists, and a sheet that prints "Owners / Directors" once has answered
    # both questions with the same rows. The percentage was stashed on the
    # director row only to be read here — the director group has no ownership
    # column, so the key is removed rather than passed through.
    promoted: List[Dict[str, str]] = []
    for row in out.get("directors_and_owners", []):
        pct = row.pop("ownership_percentage", "")
        if not pct:
            continue
        promoted.append({
            "name": row.get("name", ""),
            "nationality": row.get("nationality", ""),
            "ownership_percentage": pct,
        })
    if promoted:
        out.setdefault("ubos", []).extend(promoted)
    return out


def extract_company_fields(
    text: str,
    sections: Optional[Dict[str, List[Dict[str, str]]]] = None,
) -> List[ExtractedField]:
    """Read company/registration/bank details out of a document's text.

    Returns the scalar fields it could place, best-effort and in catalogue
    order. A field that is not confidently present is simply absent from the
    result; the officer fills those by hand.

    If `sections` (a mutable dict) is supplied, the repeatable row groups —
    directors/owners, UBOs, client references, products — are additionally
    written into it as lists of row dicts. They cannot ride along in the flat
    return value, which holds one value per key.
    """
    # One vendor per form. A multi-record sheet is cut to the record it leads
    # with before anything is read, so the whole-function passes below — IBAN,
    # TRN, country inference — cannot reach past it either.
    text = _first_scenario(text)
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
            if field in (
                "trade_license_expiry", "date_of_incorporation",
                "authorized_representative_id_expiry", "date_signed",
            ):
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
            if field in _PERSON_NAME_FIELDS:
                value = value.split(" (", 1)[0].strip()
            offer(field, value, line, EXACT)

    # ---- sub-fields packed beside a labelled value -----------------------
    # A data sheet prints one row per field, so a compound row carries several
    # fields at once: "Reg & Trade Licence CN-1029384 | Expiry: 15-Nov-2027 |
    # TRN: 100392817400003". The label table can only ever return the row's own
    # value, so the sub-fields are read here, line by line.
    for line in lines:
        for field, value in _compound_values(line):
            if field in ("trade_license_expiry",):
                iso = normalise_date(value)
                if iso:
                    offer(field, iso, line, EXACT)
                continue
            if field == "vat_registration_no":
                digits = re.sub(r"\D", "", value)
                m = _TRN_RE.search(digits)
                if m:
                    offer(field, m.group(0), line, EXACT)
                continue
            if field == "bank_iban":
                iban = normalise_iban(value)
                if iban:
                    offer(field, iban, line, EXACT)
                continue
            if field == "swift_code" and not re.fullmatch(r"[A-Za-z0-9]{8,11}", value):
                continue
            offer(field, value, line, EXACT)

    # ---- turnover printed as a dated series ------------------------------
    for line in lines:
        for field, value in _turnover_series(line):
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

    # ---- Yes/No, select-key and numeric normalisation --------------------
    # The wizard's Select3 fields store the literal strings "Yes"/"No", its
    # <select> fields store the option *key* (e.g. "goods", "simple"), and its
    # number inputs want bare digits. A document prints the human phrase, so a
    # raw capture would leave the select blank or the risk engine mis-keyed.
    # Each map is a closed vocabulary: anything unmatched is left as captured
    # rather than guessed, which is the module's standing rule.
    def _canon(field: str, raw: str) -> Optional[str]:
        v = _clean_value(raw).strip()
        low = v.lower().rstrip(".,")
        if field in _YES_NO_FIELDS:
            if low in ("yes", "y", "true", "confirmed", "applicable"):
                return "Yes"
            if low in ("no", "n", "false", "not applicable", "none"):
                return "No"
            return None
        table = _SELECT_KEYS.get(field)
        if table:
            return table.get(low)
        if field in ("turnover_year_1", "turnover_year_2", "turnover_year_3",
                     "number_of_employees"):
            digits = re.sub(r"[^0-9]", "", v)
            return digits or None
        return v

    for field in _CANONICAL_FIELDS:
        cur = found.get(field)
        if cur is None:
            continue
        canon = _canon(field, cur.value)
        if canon and canon != cur.value:
            found[field] = replace(cur, value=canon)

    # ---- repeatable row sections ----------------------------------------
    # Two shapes are collected: the per-line rows a form prints ("Director:
    # Name — Role — Country") and the one-numbered-line-per-group rows a data
    # sheet prints ("Owners / Directors 1. Name (Role, Country, 60%) | 2. …").
    # They concatenate, so a document that uses both keeps every row.
    if sections is not None:
        merged = _extract_list_sections(lines)
        for key, rows in _compound_sections(lines).items():
            merged.setdefault(key, []).extend(rows)
        sections.update(merged)

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
