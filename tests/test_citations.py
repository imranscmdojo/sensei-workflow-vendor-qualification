"""
Citation verification: what counts as grounded, and what must never be
presented as though it were.

The failure mode this file exists to prevent is a quote that is really just the
model's own sentence wearing a document name. Everything here is offline.
"""

import unittest

import citations


POLICY_CHUNK = {
    "uri": "https://drive.google.com/file/d/1POLICY/view",
    "text": (
        "Risk Scoring Matrix The Group assigns a weighted risk score to each "
        "Vendor on a scale of 1.00 to 3.00. The proposed annual spend of the "
        "contract is a factor in assessing Vendor risk, as is the country of "
        "incorporation and the nature of the goods or services supplied."
    ),
    "score": 0.68,
}

CONDUCT_CHUNK = {
    "uri": "https://drive.google.com/file/d/1CONDUCT/view",
    "text": (
        "Supplier Code of Conduct Suppliers shall not offer any inducement to "
        "a procurement employee, and shall declare any conflict of interest "
        "arising from a personal or business relationship with a bidder."
    ),
    "score": 0.55,
}


def _chunks(*chunks):
    return list(chunks)


def _assertion(rule, **extra):
    return {"section": "Risk Scoring Matrix", "rule_applied": rule, **extra}


class GroundingTest(unittest.TestCase):
    def test_a_rule_restating_policy_text_is_verified_with_a_quote(self):
        result = citations.build_citations(
            _chunks(POLICY_CHUNK),
            [_assertion(
                "The country of incorporation and the nature of the goods or "
                "services supplied are factors in assessing the risk of the "
                "Vendor.")],
            "Procurement Policy",
        )
        self.assertTrue(result[0].verified)
        self.assertIn("Risk Scoring Matrix", result[0].quote)
        self.assertEqual(result[0].source_uri, POLICY_CHUNK["uri"])

    def test_an_assertion_citing_its_own_section_heading_is_verified(self):
        """The section-heading path: even a loosely worded rule is grounded if
        it names a section the retrieved chunk is a slice of."""
        result = citations.build_citations(
            _chunks(POLICY_CHUNK),
            [_assertion("Spend above two million is always escalated.")],
            "Procurement Policy",
        )
        self.assertTrue(result[0].verified)
        self.assertEqual(result[0].source_uri, POLICY_CHUNK["uri"])

    def test_an_assertion_the_corpus_does_not_support_is_not_verified(self):
        result = citations.build_citations(
            _chunks(CONDUCT_CHUNK),
            [_assertion(
                "A vendor domiciled in a High-Risk Jurisdiction is always "
                "classified as High Risk and subject to Enhanced Due Diligence.")],
            "Procurement Policy",
        )
        self.assertFalse(result[0].verified)
        self.assertEqual(result[0].quote, "")
        self.assertEqual(result[0].source_uri, "")

    def test_no_retrieved_text_means_nothing_is_verified(self):
        result = citations.build_citations(
            [], [_assertion("Anything at all.")], "Procurement Policy")
        self.assertFalse(result[0].verified)

    def test_a_quote_is_never_invented_for_an_unverified_citation(self):
        result = citations.build_citations(
            _chunks(CONDUCT_CHUNK),
            [_assertion("Weighted spend scoring puts this vendor at 2.5.")],
            "Procurement Policy",
        )
        self.assertEqual(result[0].quote, "")


class OneChunkManyRulesTest(unittest.TestCase):
    """A live run verified only 1 of 14 citations. The cause was that each
    chunk was consumed by the first assertion that matched it, so once the Risk
    Scoring Matrix paragraph was spent every later rule fell through to
    unverified. Retrieval returns whole sections, and a section is legitimately
    the evidence for several of the risk factors."""

    RULES = [
        "The country of incorporation and the nature of the goods or services "
        "supplied are factors in assessing the risk of the Vendor.",
        "The proposed annual spend of the contract is a factor in assessing "
        "Vendor risk.",
        "The Group assigns a weighted risk score to each Vendor on a scale of "
        "1.00 to 3.00.",
    ]

    def test_one_policy_section_can_ground_several_distinct_rules(self):
        result = citations.build_citations(
            _chunks(POLICY_CHUNK),
            [_assertion(rule) for rule in self.RULES],
            "Procurement Policy",
        )
        self.assertEqual(len(result), 3)
        self.assertEqual([c.verified for c in result], [True, True, True])
        for citation in result:
            self.assertEqual(citation.source_uri, POLICY_CHUNK["uri"])

    def test_the_summary_counts_every_grounded_rule(self):
        summary = citations.citation_summary(
            citations.build_citations(
                _chunks(POLICY_CHUNK),
                [_assertion(rule) for rule in self.RULES],
                "Procurement Policy"))
        self.assertEqual(summary["verified"], 3)
        self.assertEqual(len(summary["sections"]), 1)


class CrossDocumentTest(unittest.TestCase):
    def test_each_rule_is_matched_to_the_chunk_that_actually_says_it(self):
        """The corpus holds more than the Procurement Policy, so an overlap
        match has to pick the right document rather than the first plausible
        one."""
        result = citations.build_citations(
            _chunks(CONDUCT_CHUNK, POLICY_CHUNK),
            [
                {"rule_applied": "Suppliers shall not offer any inducement to "
                                 "a procurement employee."},
                {"rule_applied": "The Group assigns a weighted risk score to "
                                 "each Vendor on a scale of 1.00 to 3.00."},
            ],
            "Procurement Policy",
            section_fallback="Unverified",
        )
        self.assertEqual([c.verified for c in result], [True, True])
        self.assertEqual(result[0].source_uri, CONDUCT_CHUNK["uri"])
        self.assertEqual(result[1].source_uri, POLICY_CHUNK["uri"])

    def test_a_quote_never_carries_a_uri_the_matcher_did_not_use(self):
        for rule, chunks in (
            ("Suppliers shall not offer any inducement to a procurement "
             "employee.", (CONDUCT_CHUNK, POLICY_CHUNK)),
            ("The proposed annual spend of the contract is a factor in "
             "assessing Vendor risk.", (CONDUCT_CHUNK, POLICY_CHUNK)),
        ):
            with self.subTest(rule=rule[:40]):
                result = citations.build_citations(
                    _chunks(*chunks), [{"rule_applied": rule}],
                    "Procurement Policy", section_fallback="Unverified")
                citation = result[0]
                allowed = {c["uri"] for c in chunks}
                if citation.verified:
                    self.assertIn(citation.source_uri, allowed)
                else:
                    self.assertEqual(citation.source_uri, "")


class QuoteMustCarryTheEvidenceTest(unittest.TestCase):
    """A live run marked three unrelated rules verified against the same chunk
    and quoted the same opening lines of it, which turned out to be an
    onboarding approval table. The matcher scored the chunk on vocabulary spread
    across the whole slice, then the quote showed only its first characters. The
    quoted span has to be where the shared terms are.

    A second live run then matched a claim about the vendor's ownership to a
    "Confirmed Sanctions Match" passage, because both mentioned a vendor, a
    risk and a requirement. Retrieval putting the subject somewhere in the
    document is not the same as the quoted text supporting the claim.
    """

    SANCTIONS_CHUNK = {
        "uri": "https://drive.google.com/file/d/1POLICY/view",
        "text": (
            "Confirmed Sanctions Match On a confirmed sanctions match, the MLRO "
            "shall, without delay and without tipping-off the Vendor: (i) block "
            "the transaction, (ii) freeze the relevant account, and (iii) "
            "escalate the matter to the Group Compliance and Internal Controls "
            "Department for a decision under Stage 1. The Vendor shall be "
            "informed only once instructed to do so by the MLRO. Any risk "
            "arising from a sanctions exposure must be recorded in the vendor "
            "file and reviewed at each periodic refresh."
        ),
        "score": 0.74,
    }

    def test_a_claim_is_not_grounded_in_a_different_rule_from_the_same_policy(self):
        """Every retrieval over one corpus returns the same background words.
        Counting those towards a match is what made a sanctions passage support
        an ownership claim."""
        result = citations.build_citations(
            [self.SANCTIONS_CHUNK],
            [{
                "section": "Risk Factors",
                "rule_applied": (
                    "The vendor's ownership and control structure is domestic "
                    "and layered, with two ultimate beneficial owners."),
            }],
            "Procurement Policy",
            section_fallback="Risk Factors",
        )
        self.assertFalse(result[0].verified,
                         msg="a sanctions passage was cited for an ownership claim")

    def test_a_genuine_match_in_the_same_document_still_verifies(self):
        """Tightening the check must not throw away real evidence."""
        result = citations.build_citations(
            [self.SANCTIONS_CHUNK, {
                "uri": "https://drive.google.com/file/d/2POLICY/view",
                "text": (
                    "Risk Factors The ownership and control structure of a "
                    "Vendor is a weighted risk factor. A layered structure with "
                    "several ultimate beneficial owners scores higher than a "
                    "simple direct holding, and the beneficial owners must be "
                    "identified in the onboarding file before approval."
                ),
                "score": 0.81,
            }],
            [{
                "section": "Risk Factors",
                "rule_applied": (
                    "The vendor's ownership and control structure is domestic "
                    "and layered, with two ultimate beneficial owners."),
            }],
            "Procurement Policy",
            section_fallback="Risk Factors",
        )
        self.assertTrue(result[0].verified)
        self.assertIn("beneficial owner", result[0].quote.lower())

    def test_background_vocabulary_alone_never_grounds_a_claim(self):
        chunks = [{
            "uri": f"https://drive.google.com/file/d/{n}/view",
            "text": ("The Vendor shall manage risk in the procurement process "
                     "and the Group shall review the vendor file at each cycle."),
            "score": 0.5,
        } for n in range(4)]
        result = citations.build_citations(
            chunks,
            [{
                "section": "Risk Factors",
                "rule_applied": ("The Vendor shall manage risk in the "
                                 "procurement process."),
            }],
            "Procurement Policy",
            section_fallback="Risk Factors",
        )
        # Every term appears in every chunk, so none of them distinguishes this
        # rule from any other passage in the same retrieval.
        self.assertEqual(citations._distinctive_terms(chunks), set())
        self.assertFalse(result[0].verified)

    def test_a_table_quote_lands_on_the_matching_row(self):
        """The policy states its category rules as a table, extracted as one
        line whose rows carry no full stops. A live run cited a
        "Professional-service providers" row for a strategic-spend rule: the
        whole table counted as a single sentence."""
        table = (
            "Vendor Category Risk Treatment Vendor Category Minimum Controls "
            "Professional-service providers (lawyers, auditors, valuers, tax "
            "advisers, consultants) Standard CDD; EDD where engaged on "
            "government-facing work, regulatory matters, or sensitive data. "
            "Construction contractors, facility-management providers, and "
            "cleaning services Standard CDD; EDD where site access is granted. "
            "Strategic or high-value suppliers (annual spend at or above "
            "AED 2,000,000) Enhanced Due Diligence; annual refresh of the "
            "vendor file. IT and software suppliers Standard CDD; EDD where "
            "the supplier handles personal data."
        )
        result = citations.build_citations(
            [{"uri": "https://drive.google.com/file/d/1POLICY/view",
              "text": table, "score": 0.8}],
            [{
                "section": "Vendor Category Risk Treatment",
                "rule_applied": ("Strategic or high-value suppliers (annual "
                                 "spend at or above AED 2,000,000) are subject "
                                 "to Enhanced Due Diligence."),
            }],
            "Procurement Policy",
            section_fallback="Vendor Category Risk Treatment",
        )
        citation = result[0]
        self.assertTrue(citation.verified)
        self.assertIn("Strategic or high-value", citation.quote)
        self.assertNotIn("Professional-service providers", citation.quote)

    def test_table_rows_are_separate_evidence_units(self):
        """The quote window must stay inside one table row. Widen it enough to
        span two rows and a claim about strategic spend comes back quoted for
        the professional-services row above it."""
        table = (
            "Vendor Category Risk Treatment Vendor Category Minimum Controls "
            "Professional-service providers Standard CDD; EDD where engaged "
            "on government-facing work or sensitive data Construction "
            "contractors Standard CDD; EDD where site access is granted "
            "Strategic or high-value suppliers with annual spend at or above "
            "AED 2,000,000 require Enhanced Due Diligence and annual refresh "
            "IT and software suppliers Standard CDD; EDD where the supplier "
            "handles personal data or has privileged access to Group systems"
        )
        cases = [
            ("Professional-service providers require Enhanced Due Diligence "
             "where engaged on government-facing work.",
             "Professional-service providers"),
            ("Strategic or high-value suppliers with annual spend at or above "
             "AED 2,000,000 require Enhanced Due Diligence and annual refresh.",
             "Strategic or high-value suppliers"),
            ("IT and software suppliers handling personal data require "
             "Enhanced Due Diligence.", "IT and software suppliers"),
        ]
        for rule, expected_row in cases:
            with self.subTest(rule=rule[:40]):
                quote = citations._quote_supporting(rule, table)
                self.assertIn(expected_row, quote)

    def test_the_window_shrinks_for_a_short_chunk(self):
        self.assertEqual(citations._window_token_width(3), 3)
        self.assertEqual(citations._window_token_width(10_000),
                         citations._QUOTE_WINDOW_TOKENS)

    def test_the_quote_not_the_chunk_decides_verification(self):
        """A chunk can contain the right sentence and still be the wrong
        evidence if the quoted span is elsewhere in it."""
        rule = ("The proposed annual spend with this vendor is AED 2,500,000 "
                "and makes the engagement a strategic supplier.")
        chunk = {
            "uri": "https://drive.google.com/file/d/1POLICY/view",
            "text": (
                "Vendor Category Risk Treatment Strategic or high-value "
                "suppliers with annual spend at or above the strategic "
                "threshold are subject to Enhanced Due Diligence and annual "
                "refresh. " + "Unrelated filler sentence about the approval "
                "workflow and the responsible department for each stage. "
                "Some other rule entirely, with no annual spend figure in it "
                "and no reference to any strategic supplier at all."
            ),
            "score": 0.7,
        }
        result = citations.build_citations(
            [chunk], [{"section": "Risk Factors", "rule_applied": rule}],
            "Procurement Policy", section_fallback="Risk Factors")
        self.assertTrue(result[0].verified)
        self.assertIn("Enhanced Due Diligence", result[0].quote)

    CHUNK = {
        "uri": "https://drive.google.com/file/d/1POLICY/view",
        "text": (
            "Compliance Onboarding approval memo; Stage Activity Responsible "
            "Output Master Vendor List entry the GCICD approves. For "
            "government-facing intermediaries, agents, and distributors, the "
            "ABAC Policy may also require GCEO approval. Address such cases in "
            "writing to the Compliance function, retaining the memo on file for "
            "audit purposes. Nothing in this passage concerns the vendor's own "
            "jurisdiction, its ownership chain, or the payment terms it has "
            "proposed. The risk factors that matter here are the country of "
            "incorporation and the structure of the ownership and control of "
            "the vendor entity, together with the proposed annual spend of the "
            "contract. Those three factors carry the most weight in the matrix."
        ),
        "score": 0.71,
    }

    RULES = [
        "The vendor's ownership and control structure is domestic and layered.",
        "The vendor is incorporated and operates in the United Arab Emirates.",
        "The proposed annual spend of the contract is AED 2,500,000.",
    ]

    def test_the_quote_never_leads_with_the_chunk_opening(self):
        """The three rules above are all evidenced by the same late passage, so
        they may legitimately share a quote. What must never happen is the quote
        being the first characters of the chunk while the evidence sits later."""
        result = citations.build_citations(
            [self.CHUNK],
            [{"rule_applied": r} for r in self.RULES],
            "Procurement Policy",
            section_fallback="Risk Factors",
        )
        for citation in result:
            if citation.verified:
                self.assertNotIn("Onboarding approval memo", citation.quote)

    def test_the_selector_follows_the_evidence_to_different_parts_of_a_chunk(self):
        """Two rules, two separate passages in one chunk: each quote must land
        on its own evidence rather than the same window for both."""
        chunk = {
            "uri": "https://drive.google.com/file/d/1POLICY/view",
            "text": (
                "Unrelated administrative boilerplate about the onboarding "
                "approval memo and the stage activity responsible for each "
                "output. Irrelevant filler to separate the two passages. "
                "The country of incorporation of the vendor and the principal "
                "operations constitute a weighted risk factor. Another "
                "sentence of filler so the two passages are far apart. "
                "Vendors that hold bearer shares with no registered holder "
                "cannot establish beneficial ownership from company records. "
                "More filler after the second passage to move it away from the "
                "start of the slice of text."
            ),
            "score": 0.7,
        }
        result = citations.build_citations(
            [chunk],
            [
                {"rule_applied": "The country of incorporation of the vendor "
                                 "is a weighted risk factor."},
                {"rule_applied": "Bearer shares mean beneficial ownership "
                                 "cannot be established from company records."},
            ],
            "Procurement Policy",
            section_fallback="Risk Factors",
        )
        first, second = (c for c in result)
        self.assertNotEqual(first.quote, second.quote)
        self.assertIn("country of incorporation", first.quote.lower())
        self.assertIn("bearer", second.quote.lower())

    def test_the_quote_contains_the_terms_that_justified_the_match(self):
        result = citations.build_citations(
            [self.CHUNK],
            [{"rule_applied": r} for r in self.RULES],
            "Procurement Policy",
            section_fallback="Risk Factors",
        )
        for citation in result:
            if not citation.verified:
                continue
            with self.subTest(rule=citation.rule_applied[:40]):
                wanted = set(citations._significant_tokens(citation.rule_applied))
                present = set(citations._significant_tokens(citation.quote))
                self.assertTrue(
                    wanted & present,
                    msg=f"quote shares no term with the rule: {citation.quote!r}",
                )

    def test_the_quote_is_not_the_opening_of_the_chunk(self):
        result = citations.build_citations(
            [self.CHUNK],
            [{"rule_applied": "The country of incorporation is a weighted risk "
                              "factor in the matrix."}],
            "Procurement Policy",
            section_fallback="Risk Factors",
        )
        citation = result[0]
        self.assertTrue(citation.verified)
        self.assertNotIn("Onboarding approval memo", citation.quote)

    def test_a_quote_with_no_supporting_terms_falls_back_rather_than_lying(self):
        """When the match came from the section heading and the prose carries
        none of the rule's terms, the passage is quoted as-is. It must still be
        text from the chunk, never a reconstruction."""
        quote = citations._quote_supporting(
            "quantum entanglement in procurement", self.CHUNK["text"])
        self.assertTrue(quote)
        self.assertIn(quote[:40], self.CHUNK["text"])

    def test_an_empty_chunk_yields_an_empty_quote(self):
        self.assertEqual(citations._quote_supporting("anything", ""), "")

    def test_a_heading_absent_from_the_chunk_is_never_borrowed(self):
        """A heading is only trustworthy if this chunk actually contains it.
        Prefixing an unchecked one would lend the authority of a section title
        to a passage the section never mentions."""
        text = "Some Vendor shall maintain insurance for the works. " \
               "Evidence shall be retained by the project file for five years."
        rule = "The vendor must retain evidence for five years in the project file."
        without = citations._quote_supporting(rule, text)
        self.assertNotIn("Risk Scoring Matrix", without)

        with_heading = citations._quote_supporting(
            rule, "Risk Scoring Matrix " + text, heading="Risk Scoring Matrix")
        self.assertIn("Risk Scoring Matrix", with_heading)

    def test_the_quote_comes_only_from_the_retrieved_chunk(self):
        result = citations.build_citations(
            [self.CHUNK],
            [{"rule_applied": r, "section": "Risk Factors"} for r in self.RULES],
            "Procurement Policy",
        )
        for citation in result:
            if not citation.verified:
                continue
            with self.subTest(rule=citation.rule_applied[:40]):
                body = citation.quote.split(". ", 1)[-1]
                self.assertIn(body[:50].rstrip(" ."), self.CHUNK["text"])


class SummaryTest(unittest.TestCase):
    def test_the_summary_counts_both_kinds_and_lists_the_sources(self):
        result = citations.build_citations(
            _chunks(POLICY_CHUNK, CONDUCT_CHUNK),
            [
                {"rule_applied": "The Group assigns a weighted risk score to "
                                 "each Vendor on a scale of 1.00 to 3.00."},
                {"rule_applied": "Vendors must hold ISO 9001 and ISO 14001."},
            ],
            "Procurement Policy",
            section_fallback="Unverified",
        )
        summary = citations.citation_summary(result)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["verified"] + summary["unverified"], 2)
        self.assertEqual(summary["verified"], 1)
        self.assertIn(POLICY_CHUNK["uri"], summary["sources"])

    def test_an_empty_result_summarises_to_zero_rather_than_raising(self):
        summary = citations.citation_summary([])
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["verified"], 0)
        self.assertEqual(summary["sources"], [])

    def test_the_cap_on_citations_is_respected(self):
        rules = [_assertion(f"Rule number {i} about vendor risk weighting.")
                 for i in range(40)]
        result = citations.build_citations(
            _chunks(POLICY_CHUNK), rules, "Procurement Policy", max_citations=12)
        self.assertEqual(len(result), 12)


class ThresholdTest(unittest.TestCase):
    def test_the_threshold_is_a_knob_not_a_constant_inside_the_matcher(self):
        rule = ("The proposed annual spend of the contract is a factor in "
                "assessing the risk of this particular Vendor over the term.")
        loose = citations.build_citations(
            _chunks(POLICY_CHUNK), [{"rule_applied": rule}], "Procurement Policy",
            section_fallback="Unverified", overlap_threshold=0.01)
        strict = citations.build_citations(
            _chunks(POLICY_CHUNK), [{"rule_applied": rule}], "Procurement Policy",
            section_fallback="Unverified", overlap_threshold=0.99)
        self.assertTrue(loose[0].verified)
        self.assertFalse(strict[0].verified)


if __name__ == "__main__":
    unittest.main()
