"""
Citation extraction from Vertex AI RAG grounding metadata.

The supplier-contract workflow asks the model to describe its own sources and
trusts the answer. That is not good enough for a compliance dossier: a citation
nobody can check is not evidence, and a model that invents a section number
produces a worse failure than one that admits it found nothing.

Grounding metadata does not need to be trusted either. When a GenerativeModel
is called with a RAG retrieval tool, the response carries
`candidates[0].grounding_metadata.grounding_chunks`, and each chunk exposes
`retrieved_context` with:

    uri   the Google Drive file the chunk was imported from
    text  the verbatim chunk text that was put in context

So a citation produced here is backed by a source file and a literal quote.
The model is only asked for the part it alone can supply: which rule in the
quote applies to this vendor.

Every citation carries `verified`, which is True only when a grounding chunk
backed it. Citations the model asserted without retrieval support are kept
but marked unverified, so a reviewer can see the difference instead of
having to trust it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Vertex RAG chunks are ~1024 tokens with 200 overlap, so the same paragraph
# commonly appears in several chunks. Quotes are clipped for display and then
# de-duplicated on their normalised text.
QUOTE_MAX_CHARS = 420

# A line this short or shorter is treated as one citable unit. The policy's
# tables are extracted as single lines whose rows are separated by spaces rather
# than newlines, so this is a ceiling on row length, not on paragraph length.
TABLE_ROW_MAX_CHARS = 220

# Token-level scanner used to map a scored window back to a character span.
# Must agree with _significant_tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Width of the sliding quote window, in tokens. Wide enough to hold a policy
# row plus its heading, narrow enough to stay inside one rule. Measured against
# the Vendor Category Risk Treatment table, which shares "enhanced due
# diligence" across every row: at 34 tokens all four rows resolve to the right
# one, and by 60 the window is wide enough to span two rows and the quote
# starts landing on the wrong one.
_QUOTE_WINDOW_TOKENS = 34

# Share of an assertion's significant terms that must also appear in a
# retrieved chunk before the assertion counts as grounded. Calibrated against
# this corpus: the retrieval pass elaborates each rule in its own words
# ("The proposed annual spend for this vendor is AED 2,500,000, which
# contributes to..."), so the denominator is the whole elaboration and the
# measured overlaps run 0.29-0.69. At 0.40 the clearly supported rules are
# verified and the ones that only share generic procurement vocabulary are not.
# Raise it to demand a near-verbatim match; lower it and the citation starts
# asserting authority the corpus never granted.
OVERLAP_THRESHOLD = 0.40

# Share of a claim's distinctive terms that must appear in the passage actually
# quoted. Separate from OVERLAP_THRESHOLD because the two answer different
# questions: that one asks whether retrieval put the claim's subject somewhere
# in the corpus, this one asks whether the text a reviewer reads supports the
# claim.
#
# Calibrated against the shared corpus on the NH-PQF-001 sample form. Phase A
# assertions scored 1.00 (category precedence), 0.69 (strategic supplier) and
# 0.68 (professional services), so 0.60 sits under the floor with margin for
# the variation between retrieval draws, while the claims that were wrongly
# verified before are now rejected upstream, at chunk selection, and never
# reach this check.
#
# Held high on purpose. A false positive here is a fabricated authority; a
# false negative only shows as an honestly labelled unverified citation.
QUOTE_OVERLAP_THRESHOLD = 0.60

# A term appearing in more than this share of the retrieved chunks is
# background vocabulary for this run, not evidence for any particular rule.
DISTINCTIVE_DF_MAX = 0.60

# Section headings in the Procurement Policy are sentence-case lines that are
# short, unpunctuated at the end, and sit above a block of prose. Extracted
# from the chunk text so the citation carries the section it came from without
# trusting the model to name it.
_HEADING_HINT = re.compile(
    r"^(?:"
    r"(?:Risk Factors|Weighted Risk Score|Risk Level|Refresh Cycle|Due Diligence|"
    r"Mandatory High-Risk Classification|Vendor Category Risk Treatment|"
    r"Minimum Controls|Vendor Category|Onboarding Integrity Rule|"
    r"Vendor Registration and Onboarding Workflow|KYC and Due Diligence[^|]*|"
    r"Prequalification and the Master Vendor List|Prequalification Criteria|"
    r"Shortlisting|Approval for Master Vendor List Entry|"
    r"ABAC Integrity Review[^|]*|Sanctions, PEP, and Adverse-Media Screening|"
    r"Confirmed Sanctions Match|Periodic Re-screening and Monitoring|"
    r"Suspension, Blacklisting, and Reinstatement|"
    r"Vendor Performance Evaluation|Sub-Contracting)"
    r")\s*$"
)

_TRAILING_TABLE_JUNK = re.compile(r"\s*(?:Stage Activity Responsible Output|"
                                   r"Vendor Category Risk Treatment Minimum Controls)\s*$")


@dataclass
class Citation:
    """One policy rule applied to the vendor, backed by retrieved text."""

    document: str
    section: str
    rule_applied: str
    quote: str = ""
    source_uri: str = ""
    score: float = 0.0
    verified: bool = False
    matched_trigger: Optional[str] = None
    factor: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document,
            "section": self.section,
            "rule_applied": self.rule_applied,
            "quote": self.quote,
            "source_uri": self.source_uri,
            "score": round(float(self.score), 4),
            "verified": self.verified,
            "matched_trigger": self.matched_trigger,
            "factor": self.factor,
        }


# --------------------------------------------------------------------------
# Grounding metadata
# --------------------------------------------------------------------------

def _iter_grounding_chunks(response: Any) -> List[Any]:
    """Pull grounding chunks off a Vertex response, tolerating the SDK's
    several response shapes (a plain response, a stream's final chunk, or a
    response whose candidates list is empty)."""
    candidates = getattr(response, "candidates", None) or []
    chunks: List[Any] = []
    for candidate in candidates:
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding is None:
            continue
        for chunk in getattr(grounding, "grounding_chunks", None) or []:
            chunks.append(chunk)
    return chunks


def _chunk_payload(chunk: Any) -> Dict[str, Any]:
    """Normalise one grounding chunk to {uri, text, score}."""
    context = getattr(chunk, "retrieved_context", None)
    uri = ""
    text = ""
    if context is not None:
        uri = str(getattr(context, "uri", "") or "")
        text = str(getattr(context, "text", "") or "")
    if not uri or not text:
        web = getattr(chunk, "web", None)
        if web is not None:
            uri = uri or str(getattr(web, "uri", "") or "")
    score = 0.0
    for candidate_score in (getattr(chunk, "scores", None) or []):
        value = getattr(candidate_score, "score", None)
        if value is not None:
            try:
                score = max(score, float(value))
            except (TypeError, ValueError):
                continue
    return {"uri": uri, "text": text, "score": score}


def grounding_chunks(response: Any) -> List[Dict[str, Any]]:
    """All retrieved chunks for a response, highest score first."""
    chunks = [_chunk_payload(c) for c in _iter_grounding_chunks(response)]
    chunks = [c for c in chunks if c["text"]]
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks


# --------------------------------------------------------------------------
# Section heading extraction
# --------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"[\r\n\t]+", " ", text or "").strip()


def extract_section_headings(chunk_text: str) -> List[str]:
    """Pull policy section headings out of a retrieved chunk.

    The chunk is a slice of a PDF, so headings are the short standalone lines
    that title a block. Returns them in order of appearance.
    """
    headings: List[str] = []
    for raw_line in (chunk_text or "").splitlines():
        line = _clean(raw_line)
        if not line or len(line) > 120:
            continue
        if _HEADING_HINT.match(line):
            headings.append(line)
            continue
        # A heading mid-paragraph is often glued to the following column, e.g.
        # "Vendor Category Risk TreatmentThe following risk-treatment matrix ..."
        match = re.match(r"^([A-Z][A-Za-z0-9 ,\-/&'()]{6,80}?)(?=[A-Z][a-z])", line)
        if match and _HEADING_HINT.match(match.group(1).strip()):
            headings.append(match.group(1).strip())
    seen: set = set()
    ordered: List[str] = []
    for heading in headings:
        if heading.lower() not in seen:
            seen.add(heading.lower())
            ordered.append(heading)
    return ordered


def _clip_quote(text: str) -> str:
    quote = _TRAILING_TABLE_JUNK.sub("", _clean(text))
    if len(quote) <= QUOTE_MAX_CHARS:
        return quote
    clipped = quote[:QUOTE_MAX_CHARS]
    boundary = max(clipped.rfind(". "), clipped.rfind(".\n"))
    if boundary > QUOTE_MAX_CHARS * 0.5:
        return clipped[: boundary + 1].strip()
    return clipped.rstrip() + "..."


def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _quote_supporting(rule: str, chunk_text: str, heading: str = "") -> str:
    """The part of the chunk that actually carries the matched terms.

    Quoting the head of the chunk is what a citation must not do. A live run
    produced three rules about ownership, jurisdiction and services of supply,
    all "verified" against the same opening lines of a chunk that was really an
    onboarding approval table: the matcher had scored the chunk on vocabulary
    spread across all of it, and the quote then showed only its first
    characters. The quote is what a reviewer reads, so it has to contain the
    evidence, not merely come from a file that contains it somewhere.

    The search is a sliding window over the text rather than a split into
    sentences. The Procurement Policy states most of its rules in tables whose
    rows are glued together with spaces and carry no full stops, so a sentence
    split keeps a whole table as one unit and quotes whichever row shares the
    most words. Splitting on capitalised words instead shatters ordinary
    sentences. A character window is indifferent to both: it just finds where in
    the text the claim's terms actually are.
    """
    text = _clean(chunk_text)
    if not text:
        return ""

    wanted = set(_significant_tokens(rule))
    if not wanted:
        return _clip_quote(text)

    # Score sliding windows by how much of the rule's *rare* vocabulary they
    # carry. Weighting each term by how often it appears in this chunk is what
    # separates two rows of the same table: "enhanced due diligence" appears in
    # every category row and so says nothing, while "strategic" and the spend
    # figure appear only in the row that is actually about strategic spend.
    tokens = list(_TOKEN_RE.finditer(text.lower()))
    if not tokens:
        return _clip_quote(text)

    surface = [t.group(0) for t in tokens]
    frequency: Dict[str, int] = {}
    for token in surface:
        if token in wanted:
            frequency[token] = frequency.get(token, 0) + 1

    if not frequency:
        # The match came from the section heading rather than the prose, so the
        # passage itself is the best available evidence.
        return _clip_quote(text)

    # A token the claim never mentions carries no weight at all. Giving it 1.0
    # made every window of the same size score identically, so the search
    # always returned the first one and never actually looked at the text.
    weights = [
        (1.0 / frequency[token]) if token in frequency else 0.0
        for token in surface
    ]

    width = _window_token_width(len(tokens))
    best_start, best_score = 0, -1.0
    running = sum(weights[:width])
    best_score = running
    for start in range(1, len(tokens) - width + 1):
        running += weights[start + width - 1] - weights[start - 1]
        if running > best_score:
            best_score, best_start = running, start

    chosen = tokens[best_start:best_start + width]
    quoted = text[chosen[0].start():chosen[-1].end()].strip()

    # Carry the section heading into the quote. A reviewer needs to see which
    # part of the policy the evidence sits under. Only a heading that is present
    # in this chunk's own text is used, so a heading sniffed out of the
    # surrounding slice cannot lend authority to an unrelated passage.
    label = _clean(heading)
    if label and _normalise(label) not in _normalise(quoted) \
            and _normalise(label) in _normalise(text):
        quoted = f"{label}. {quoted}"

    return _clip_quote(quoted)


def _window_token_width(token_count: int) -> int:
    """How many tokens a quote window spans, in tokens rather than characters.

    Capped by what is available so a short chunk is quoted whole instead of
    yielding no candidate windows at all.
    """
    if token_count <= 4:
        return token_count
    return min(_QUOTE_WINDOW_TOKENS, token_count)


# --------------------------------------------------------------------------
# Citation assembly
# --------------------------------------------------------------------------

def build_citations(
    retrieved: Iterable[Dict[str, Any]],
    assertions: Iterable[Dict[str, Any]],
    document: str,
    section_fallback: str = "Vendor Lifecycle Management",
    max_citations: int = 12,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> List[Citation]:
    """Merge retrieval chunks with the model's rule assertions.

    `retrieved`  grounding chunks from citations.grounding_chunks(response)
    `assertions` objects with at least a `rule_applied` string, optionally
                  `document`, `section`, `matched_trigger`, `factor`

    An assertion is marked verified only when the quoted passage itself carries
    the claim's distinctive content. Matching a chunk is not enough: retrieval
    returns whole policy sections, so a claim about ownership happily matches a
    chunk that is really a sanctions rule. Unverified assertions are retained
    and clearly labelled rather than dropped, so a reviewer sees that retrieval
    did not corroborate the rule.

    One chunk may back several assertions. The corpus retrieves whole policy
    sections, and a section is legitimately the evidence for more than one of the
    risk factors; consuming each chunk once would cap verification at a single
    citation for any run that reasons about a vendor factor by factor.
    """
    chunks = [c for c in retrieved if c.get("text")]
    citations: List[Citation] = []
    distinctive = _distinctive_terms(chunks)

    for assertion in assertions or []:
        if not isinstance(assertion, dict):
            continue
        rule = _clean(str(assertion.get("rule_applied") or ""))
        if not rule:
            continue

        claimed_section = _clean(str(assertion.get("section") or ""))
        matched, overlap = _best_chunk(rule, claimed_section, chunks,
                                       overlap_threshold)
        chunk = matched or {}
        quote = ""
        section = claimed_section or section_fallback
        verified = False
        score = 0.0

        if matched is not None:
            headings = extract_section_headings(chunk.get("text", ""))
            section = claimed_section or (headings[0] if headings else section_fallback)
            quote = _quote_supporting(rule, chunk.get("text", ""), heading=section)
            score = float(chunk.get("score") or 0.0)
            # The reviewer is shown the quote, not the chunk, so the quote is
            # what has to carry the claim. Retrieval returning a chunk from the
            # right document is not evidence: a live run marked a claim about
            # ownership verified against a Confirmed Sanctions Match passage
            # from the same policy.
            verified = _quote_supports(rule, quote, distinctive)
        else:
            quote = ""
            score = 0.0
            verified = False

        citations.append(Citation(
            document=_clean(str(assertion.get("document") or "")) or document,
            section=section,
            rule_applied=rule,
            quote=quote,
            source_uri=str(chunk.get("uri") or ""),
            score=score,
            verified=verified,
            matched_trigger=(str(assertion["matched_trigger"])
                             if assertion.get("matched_trigger") else None),
            factor=(str(assertion["factor"]) if assertion.get("factor") else None),
        ))

        if len(citations) >= max_citations:
            break

    return citations


def _distinctive_terms(chunks: List[Dict[str, Any]]) -> set:
    """Terms that carry meaning in *this* retrieval, not just vocabulary.

    Retrieval over one policy returns the same connective words in every chunk
    ("vendor", "risk", "procurement", "shall"). Counting those towards an
    overlap is what let a claim about ownership be "supported" by a sanctions
    passage: the shared words were the generic ones. A term that appears in
    more than `DISTINCTIVE_DF_MAX` of the retrieved chunks is treated as
    background and cannot, on its own, ground anything.
    """
    if not chunks:
        return set()
    frequency: Dict[str, int] = {}
    for chunk in chunks:
        for token in set(_significant_tokens(chunk.get("text", ""))):
            frequency[token] = frequency.get(token, 0) + 1

    ceiling = max(1, int(len(chunks) * DISTINCTIVE_DF_MAX))
    return {token for token, count in frequency.items() if count <= ceiling}


def _quote_supports(rule: str, quote: str, distinctive: set) -> bool:
    """Does the passage the reviewer will actually read carry the claim?

    Requires the quote to contain a real share of the claim's *distinctive*
    terms. Where nothing distinctive is available the rule is left unverified
    rather than credited to a passage that merely shares the surrounding
    vocabulary.
    """
    if not quote:
        return False

    wanted = set(_significant_tokens(rule))
    if not wanted:
        return False

    # Only the claim's distinctive terms can ground it. If a retrieval offers
    # none, the claim shares nothing with any passage in it but background
    # vocabulary, and there is no evidence to report. Falling back to the full
    # term set here is what re-created the false positives this check exists to
    # catch, because the generic words are always present.
    key = wanted & distinctive
    if not key:
        return False

    present = set(_significant_tokens(quote))
    return len(key & present) / len(key) >= QUOTE_OVERLAP_THRESHOLD


def _best_chunk(rule: str, claimed_section: str, chunks: List[Dict[str, Any]],
                threshold: float = OVERLAP_THRESHOLD
                ) -> Tuple[Optional[Dict[str, Any]], float]:
    """Find the retrieved chunk that best supports a rule assertion.

    Returns the chunk and the overlap that justified it, so the caller can tell
    a marginal match from a solid one.
    """
    if not chunks:
        return None, 0.0

    if claimed_section:
        heading_key = _normalise(claimed_section)
        for chunk in chunks:
            if heading_key and heading_key in _normalise(chunk.get("text", "")):
                return chunk, 1.0

    rule_tokens = set(_significant_tokens(rule))
    if not rule_tokens:
        return None, 0.0

    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for chunk in chunks:
        chunk_tokens = set(_significant_tokens(chunk.get("text", "")))
        if not chunk_tokens:
            continue
        overlap = len(rule_tokens & chunk_tokens) / len(rule_tokens)
        if overlap > best_score:
            best_score = overlap
            best = chunk

    # Below this overlap the "citation" is really just the model's own words
    # with a document name attached, so it is reported as unverified instead.
    return (best, best_score) if best_score >= threshold else (None, best_score)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at", "by",
    "with", "from", "is", "are", "be", "been", "being", "that", "this", "these",
    "those", "it", "its", "as", "any", "all", "not", "but", "if", "than", "then",
    "so", "such", "which", "who", "whom", "whose", "what", "when", "where",
    "will", "would", "shall", "should", "must", "may", "can", "cannot", "do",
    "does", "did", "has", "have", "had", "was", "were", "been", "per", "each",
    "every", "no", "nor", "only", "also", "more", "most", "other", "others",
}


def _significant_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    meaningful = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return meaningful or [t for t in tokens if len(t) > 2]# --------------------------------------------------------------------------
# Policy reference
# --------------------------------------------------------------------------

def policy_citations(citations: List[Citation]) -> List[Dict[str, Any]]:
    """The `rag_retrieval_citations` array from output_schema.md, in the exact
    shape the contract with Tool 2 requires, with the verification evidence
    attached."""
    return [c.to_dict() for c in citations]


def citation_summary(citations: List[Citation]) -> Dict[str, Any]:
    verified = sum(1 for c in citations if c.verified)
    return {
        "total": len(citations),
        "verified": verified,
        "unverified": len(citations) - verified,
        "sources": sorted({c.source_uri for c in citations if c.source_uri}),
        "sections": sorted({c.section for c in citations if c.section}),
    }
