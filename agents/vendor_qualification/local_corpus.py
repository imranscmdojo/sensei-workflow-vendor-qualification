"""The local policy store — Phase A retrieval when the Vertex corpus is silent.

Phase A asks the Vertex RAG corpus for the policy rules that bear on a
submission. That pass can come back empty for reasons that have nothing to do
with the vendor: the corpus endpoint unreachable, a response that never
conforms, a run with no grounding chunks. The dossier then carries no
citations at all — "No policy passages were retrieved" — on an assessment that
did in fact apply the policy.

This module is the fallback retrieval pass over a small, curated copy of the
governing sections held in-repo. It is still retrieval, not generation:

- every passage below is policy text, stored verbatim and quoted verbatim;
- every rule is a sentence taken from the passage that carries it, so
  `citations.build_citations` marks it verified against real text in the
  normal way (section heading in the chunk, quote window over the same
  words);
- the dossier records which store served the pass under `retrieval_metadata`,
  so a reviewer can always tell a local-store citation from a corpus one.

The store is deliberately small: the band table, the weighted factors, the
mandatory high-risk list, the category treatment, the prequalification
criteria, the Appendix F sheet, the Form 73/74 verification rules, the
Spec 3.4 / Assumption A7 intake rules, and the Stage 3 screening rule. Those
are the sections the qualification decision is made of, and a prequalification
run that cites none of them is an assessment with no visible basis.

The country Risk List is deliberately absent. A jurisdiction tier must come
from the policy's published list or be reported as undetermined (see the
retrieval discipline in the system instructions), and this store does not
carry the list — so `jurisdiction_tier` comes back empty and the engine keeps
the form's declared tier, exactly as it does when retrieval is silent about a
country.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SOURCE = "local-policy-corpus"
CORPUS_LABEL = "National Holding Procurement Policy (local store)"

# uri prefix for a passage; the slug doubles as the anchor the reviewer can
# quote from ("local://procurement-policy/<slug>").
_URI_PREFIX = "local://procurement-policy/"


class Passage:
    """One policy section: heading, stored text, and the rule quoted from it.

    `rule` must be a sentence that appears in `text` and should stay under
    ~30 tokens: the citation's quote window spans 34 tokens, so a rule longer
    than the window can fail the 0.60 quote-overlap check and be reported
    unverified even though its own passage carries it.
    """

    __slots__ = ("section", "slug", "text", "rule", "matched_trigger", "factor")

    def __init__(self, section: str, slug: str, text: str, rule: str,
                 matched_trigger: str = "", factor: str = "") -> None:
        self.section = section
        self.slug = slug
        self.text = text
        self.rule = rule
        self.matched_trigger = matched_trigger
        self.factor = factor


PASSAGES: List[Passage] = [
    # First in the store on purpose: `_best_chunk`'s heading scan returns the
    # first passage whose text contains the claimed heading, and the matrix
    # below cross-references "mandatory high-risk classification" by name.
    Passage(
        section="Mandatory High-Risk Classification",
        slug="mandatory-high-risk-classification",
        text=(
            "Mandatory High-Risk Classification. Regardless of the weighted "
            "score, an unresolved sanctions match not adjudicated as a false "
            "positive by the MLRO is always High Risk and subject to Enhanced "
            "Due Diligence. The following are likewise always High Risk and "
            "subject to Enhanced Due Diligence: a PEP, PEP family member or "
            "close associate; a counterparty domiciled, incorporated, "
            "operating or beneficially owned in a High-Risk Jurisdiction; a "
            "complex, multi-layered, offshore, nominee, trust, foundation or "
            "bearer-share structure; a third-party intermediary, agent, "
            "distributor, consultant or lobbyist engaged to interact with "
            "government officials; a joint-venture partner, co-investor or "
            "M&A target; a credible adverse-media association with financial "
            "crime, corruption, terrorism, human-rights abuses or organised "
            "crime; or a GCICD determination that heightened risk warrants "
            "EDD."
        ),
        rule=(
            "Regardless of the weighted score, an unresolved sanctions match not "
            "adjudicated as a false positive by the MLRO is always High Risk and "
            "subject to Enhanced Due Diligence."
        ),
        matched_trigger="UNRESOLVED_SANCTIONS_MATCH",
        factor="",
    ),
    Passage(
        section="Weighted Risk Score / Risk Level / Due Diligence / Refresh Cycle",
        slug="risk-scoring-matrix",
        text=(
            "Weighted Risk Score / Risk Level / Due Diligence / Refresh Cycle. "
            "The Risk Scoring Matrix bands the weighted risk score (1.00 - 3.00) "
            "as follows: 1.00 - 1.60 is Low risk, Simplified Due Diligence (SDD), "
            "refreshed every 3 years; 1.61 - 2.20 is Medium risk, Standard "
            "Customer Due Diligence (CDD), refreshed every 2 years; 2.21 - 3.00 "
            "is High risk, Enhanced Due Diligence (EDD), refreshed annually. The "
            "EDD threshold is a weighted score of 2.21 or above, after any "
            "mandatory high-risk classification has been applied."
        ),
        rule=(
            "The EDD threshold is a weighted score of 2.21 or above, after any "
            "mandatory high-risk classification has been applied."
        ),
        factor="",
    ),
    Passage(
        section="Risk Factors",
        slug="risk-factors",
        text=(
            "Risk Factors. The weighted risk score is built from eight weighted "
            "factors: ownership and control structure; jurisdiction of "
            "incorporation and principal operations; nature and sensitivity of "
            "the goods or services; annual spend and strategic importance; "
            "exposure to government interaction or licensing; proposed payment "
            "structure including success fees and commissions; reputational "
            "indicators from adverse-media screening; and prior relationship "
            "history with the Group. Each factor is scored against the vendor's "
            "declared signals and the published weights are applied by the "
            "scoring engine."
        ),
        rule=(
            "Each factor is scored against the vendor's declared signals and the "
            "published weights are applied by the scoring engine."
        ),
        factor="",
    ),
    Passage(
        section="Vendor Category Risk Treatment",
        slug="vendor-category-risk-treatment",
        text=(
            "Vendor Category Risk Treatment. Where a vendor falls into more than "
            "one category, the stricter treatment applies. Government-facing "
            "intermediaries, agents, consultants and lobbyists are Always EDD, "
            "with written justification for engagement, anti-bribery undertakings "
            "and audit rights in the contract, ABAC training of the "
            "intermediary, payment only against approved deliverables, "
            "pre-approval of gifts and hospitality, and GCEO approval for "
            "onboarding. Distributors and resellers are Always EDD with ABAC and "
            "sanctions representations, end-customer screening, audit rights and "
            "periodic re-screening. Strategic or high-value suppliers with "
            "annual spend of AED 2,000,000 or more take Standard CDD with EDD on "
            "risk indicators and annual performance evaluation. Routine "
            "suppliers of non-sensitive goods and services take SDD or Standard "
            "CDD by score."
        ),
        rule=(
            "Strategic or high-value suppliers with annual spend of AED 2,000,000 "
            "or more take Standard CDD with EDD on risk indicators."
        ),
        factor="annual_spend",
    ),
    Passage(
        section="Prequalification Criteria",
        slug="prequalification-criteria",
        text=(
            "Prequalification Criteria. Stage 2 prequalification requires at "
            "least three major client references, turnover assessed against the "
            "proposed spend rather than in isolation, a declaration of whether "
            "the financial statements are independently audited, and the "
            "mandatory supporting documents for Items 60 to 81 that apply to the "
            "supplier. A reference set shorter than three is a fail: fewer than "
            "three client references does not pass prequalification."
        ),
        rule=(
            "A reference set shorter than three is a fail: fewer than three "
            "client references does not pass prequalification."
        ),
        factor="",
    ),
    Passage(
        section="Appendix F - Prequalification Scoring Sheet",
        slug="appendix-f-scoring-sheet",
        text=(
            "Appendix F - Prequalification Scoring Sheet. The Stage 5 "
            "prequalification score sheet scores three categories on a raw 0-100 "
            "scale: financial standing (weight 35), technical capability and "
            "project experience (weight 35), and ISO / HSE and quality "
            "compliance (weight 30). The weighted total passes at 70.0 of 100, "
            "and no category may sit below half of its own maximum, so a strong "
            "technical score cannot carry a financially unqualified vendor."
        ),
        rule=(
            "The weighted total passes at 70.0 of 100, and no category may sit "
            "below half of its own maximum."
        ),
        factor="",
    ),
    Passage(
        section="Form 73 / Form 74 Verification Requirements",
        slug="form-73-74-verification",
        text=(
            "Form 73 / Form 74 Verification Requirements. Where supplies exceed "
            "or are expected to exceed AED 375,000 in twelve months, a written "
            "confirmation from an authorised UAE bank (Form 73) is required. "
            "Where the registered or operating address changed more than twice "
            "in the previous twelve months, evidence explaining the changes "
            "(Form 74) is required. The Form 68 bank details confirmation letter "
            "is verified against the account declared on Form 73, and the Part 2 "
            "VAT and bank threshold checks at AED 100,000 and AED 375,000 are "
            "internal controls, not supplier declarations."
        ),
        rule=(
            "Where supplies exceed or are expected to exceed AED 375,000 in "
            "twelve months, a written confirmation from an authorised UAE bank "
            "(Form 73) is required."
        ),
        factor="annual_spend",
    ),
    Passage(
        section="Spec 3.4 - Invitation, Chasing and Assumption A7",
        slug="spec-3-4-assumption-a7",
        text=(
            "Spec 3.4 - Invitation, Chasing and Assumption A7. Spec 3.4 issues "
            "the supplier invitation on day 0 as a secure link with a 14-day "
            "lifetime, chases the outstanding form on day 3 with the invitation "
            "contacts copied, and escalates to the inviter on day 7. "
            "Assumption A7: an unanswered clarification or outstanding "
            "mandatory field is treated as absent — never assumed, never "
            "scored — and a reply that does not state a value does not supply "
            "one."
        ),
        rule=(
            "Assumption A7: an unanswered clarification or outstanding mandatory "
            "field is treated as absent — never assumed, never scored — and a "
            "reply that does not state a value does not supply one."
        ),
        factor="",
    ),
    Passage(
        section="Sanctions, PEP and Adverse-Media Screening",
        slug="sanctions-pep-adverse-media-screening",
        text=(
            "Sanctions, PEP and Adverse-Media Screening. Stage 3 screening of "
            "the vendor, its owners and its authorised representatives is "
            "adjudicated by the Subsidiary Compliance Officer and the MLRO. An "
            "unresolved sanctions match blocks prequalification rather than "
            "escalating it: the engagement may not proceed to onboarding until "
            "the MLRO records an adjudication, and retained vendors are "
            "re-screened periodically."
        ),
        rule=(
            "An unresolved sanctions match blocks prequalification rather than "
            "escalating it: the engagement may not proceed to onboarding until "
            "the MLRO records an adjudication."
        ),
        matched_trigger="UNRESOLVED_SANCTIONS_MATCH",
        factor="adverse_media",
    ),
]


def retrieve(form: Optional[Dict[str, Any]] = None,
             signal_bag: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The Phase A fallback: every governing passage, with one rule each.

    Returns the same shape the Vertex pass produces — `retrieval` (RETRIEVAL_SCHEMA
    keys plus a `_meta` provenance block) and `chunks` (uri/text/score) — so the
    caller cannot tell the two sources apart by shape, only by the metadata
    that deliberately records which store answered.
    """
    form = form if isinstance(form, dict) else {}

    chunks: List[Dict[str, Any]] = []
    rules: List[Dict[str, Any]] = []
    for index, passage in enumerate(PASSAGES):
        chunks.append({
            "uri": _URI_PREFIX + passage.slug,
            "text": passage.text,
            "score": round(1.0 - index * 0.01, 2),
        })
        rules.append({
            "section": passage.section,
            "rule_applied": passage.rule,
            "matched_trigger": passage.matched_trigger,
            "factor": passage.factor,
        })

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    retrieval: Dict[str, Any] = {
        # Deliberately empty: this store carries no country Risk List, so the
        # tier is not established here. The engine keeps the form's declared
        # tier, which is the honest outcome — see the module docstring.
        "jurisdiction_tier": "",
        "jurisdiction_basis": "",
        "business_category": "Undetermined",
        "rules": rules,
        "_meta": {
            "source": SOURCE,
            "corpus": CORPUS_LABEL,
            "retrieved_at": retrieved_at,
            "passages": len(chunks),
        },
    }
    _ = form  # selection is the whole rule book today; the form pins the seam
    return {"retrieval": retrieval, "chunks": chunks}
