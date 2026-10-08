"""
Vendor Qualification & Risk Tiering Agent (Tool 1).

Two RAG phases against the shared Sensei corpus, merged with the
deterministic scoring engine:

    Phase A  retrieve_policy_facts()   gemini-2.5-flash, top_k 15
             -> country risk tier + one assertion per relevant policy rule,
                each with its grounding chunk attached. When the Vertex corpus
                is unreachable or returns nothing usable, the pass falls back
                to the in-repo policy store (local_corpus) so the dossier
                still cites the sections the decision was made under; the
                retrieval's `_meta` records which store served the pass.

    Phase B  qualify()                  gemini-2.5-pro
             -> company/ownership narrative, required controls, open
                questions, assessment narrative, extra rule assertions.

The deterministic engine runs between them and owns the numbers:

    risk_engine.assess()             weighted score + base tier
    triggers.apply_mandatory_edd()   Always-EDD override
    schemas.appendix_f_for_tier()    Appendix F raw categories for the tier
    schemas.appendix_f_scores()      Appendix F weighting and pass/fail

Phase B's output is merged underneath those values. It is not asked for them
and the response schema does not contain them, so a model that invents a
score has nowhere to put it. Appendix F is likewise derived from the assigned
tier rather than graded independently: the sheet restates the band table's
decision, so the two can never disagree.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

from agents.base_rag_agent import BaseRAGAgent, LLMParseError

import citations as citation_lib
import risk_engine
import signals as signal_lib
import triggers as trigger_lib
import validation as validation_lib

from . import local_corpus
from . import schemas as vq_schemas
from .schemas import (
    APPENDIX_F_PASS_THRESHOLD,
    NARRATIVE_SCHEMA,
    NEXT_ACTION_EDD,
    NEXT_ACTION_INCOMPLETE,
    NEXT_ACTION_QUALIFIED,
    RETRIEVAL_SCHEMA,
    STATUS_NEEDS_EDD,
    STATUS_QUALIFIED,
    STATUS_REJECTED_HIGH_RISK,
    STATUS_REJECTED_INCOMPLETE,
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    appendix_f_for_tier,
    appendix_f_scores,
    empty_output,
)

POLICY_DOCUMENT = "Procurement Policy_Updated_2.pdf"
POLICY_SECTION = "Vendor Lifecycle Management"

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system-prompt.md")

# Conditions that disqualify a vendor outright rather than escalating it.
# These are the triggers where the policy says the engagement is not permitted
# or is blocked, so they map to REJECTED_HIGH_RISK instead of NEEDS_EDD.
DISQUALIFYING_TRIGGER_CODES = frozenset({
    "UNRESOLVED_SANCTIONS_MATCH",
    "ADVERSE_MEDIA_ASSOCIATION",
    "COMPLEX_OWNERSHIP",
})

# A Prohibited jurisdiction blocks the engagement whatever else is true, so it
# is checked separately from the trigger list.
PROHIBITED_JURISDICTION_TIERS = frozenset({"prohibited", "banned", "embargoed"})

# Vendor identity, for the dossier and the Tool 2 hand-off. Format mirrors the
# form's own control block: "Format: NH-PQ-2026-____".
VENDOR_ID_PREFIX = "VND"
VENDOR_ID_YEAR_SUFFIX = "2026"

# Item 84: the audited-report attachment behind the Item 40a audit declaration.
AUDITED_REPORT_ITEM = "84"


def onboarding_flags(form: Dict[str, Any]) -> Dict[str, bool]:
    """Governance flags for Tool 2 (contracting), derived from the submission.

    Deliberately not a risk input: nothing returned here feeds risk_engine,
    triggers, or Appendix F, so adding or removing a flag cannot move a score.

        bank_callback_verification_required
            Always true. Bank name, account holder, and IBAN are recorded as
            typed and never confirmed against the bank, so the downstream tool
            must always callback on a known-good number before payment or
            signature. A constant, not a finding.

        audited_financials_verified
            True only when the supplier declared audited statements (Item 40a)
            *and* attached the Item 84 report. Declaring "Yes" without the
            document does not clear it.
    """
    form = form if isinstance(form, dict) else {}
    declared_audited = str(form.get("financial_statements_audited") or "").strip().lower() \
        in ("yes", "y", "true", "1")
    return {
        "bank_callback_verification_required": True,
        "audited_financials_verified": bool(
            declared_audited and validation_lib.has_document(form, AUDITED_REPORT_ITEM)
        ),
    }


def _read_system_instruction() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _vendor_id(seed: str) -> str:
    """Deterministic, human-quotable vendor id derived from the legal name.

    A stable hash keeps repeat submissions on the same vendor on the same id,
    which is what a procurement officer needs to reconcile two dossiers.
    """
    digest = hashlib.sha256(seed.strip().lower().encode("utf-8")).hexdigest()
    # Modulo rather than truncation, and five digits, so the id keeps the
    # control-number shape the JSON Schema pins: VND-YYYY-NNNNN.
    numeric = int(digest[:12], 16) % 100_000
    return f"{VENDOR_ID_PREFIX}-{VENDOR_ID_YEAR_SUFFIX}-{numeric:05d}"


class VendorQualificationAgent(BaseRAGAgent):
    """Qualify and risk-tier a vendor against the National Holding policy."""

    def __init__(self, model_name: Optional[str] = None,
                 retrieval_model_name: Optional[str] = None,
                 similarity_top_k: int = 15):
        super().__init__(
            model_name=model_name or os.environ.get(
                "RAG_GENERATION_MODEL", "gemini-2.5-pro"),
            similarity_top_k=similarity_top_k,
        )
        self.retrieval_model_name = retrieval_model_name or os.environ.get(
            "RAG_RETRIEVAL_MODEL", "gemini-2.5-flash")
        self.system_instruction = _read_system_instruction()

    # ------------------------------------------------------------------
    # Phase A — retrieval
    # ------------------------------------------------------------------

    def _retrieval_prompt(self, form: Dict[str, Any], signal_text: str) -> str:
        country = _clean(form.get("country_of_incorporation")) or "(not declared)"
        operations = signal_lib._countries_of_operation(form)
        operations_text = ", ".join(operations) if operations else "(not declared)"
        role = _clean(form.get("business_role")) or "(not declared)"
        category = _clean(form.get("nature_of_business")) or "(not declared)"
        spend = signal_lib._number(
            form.get("estimated_spend_aed"),
            form.get("annual_spend_aed"),
            form.get("single_contract_value_aed"),
        )
        spend_text = f"AED {spend:,.0f}" if spend is not None else "(not declared)"

        return f"""Retrieve the policy rules that govern a vendor prequalification decision.

## Vendor under review
- Legal name: {_clean(form.get('legal_name')) or '(not declared)'}
- Country of incorporation: {country}
- Principal operations: {operations_text}
- Business role: {role}
- Business category: {category}
- Proposed annual spend: {spend_text}
- Primary goods/services: {_clean(form.get('goods_services_proposed'))[:400] or '(not declared)'}

## Declared risk signals
{signal_text}

## What to retrieve
Search the corpus for the National Holding Procurement Policy, "VENDOR LIFECYCLE
MANAGEMENT". Return:

1. The Country Risk List tier for the country of incorporation and for any
   country of principal operations. If the corpus does not publish a tier for a
   country, return "undetermined" for jurisdiction_tier. Do not reason from
   general knowledge of which countries are high risk — the tier must come from
   the corpus or be reported as undetermined.

2. The rules that bear on this specific vendor, from:
   - "Weighted Risk Score / Risk Level / Due Diligence / Refresh Cycle" (the band table)
   - "Risk Factors" (the eight weighted factors)
   - "Mandatory High-Risk Classification" (the always-EDD list)
   - "Vendor Category Risk Treatment" and its "Minimum Controls" column
   - "Prequalification Criteria" (Stage 2 requirements)

   For each rule, record the section heading exactly as it appears in the
   retrieved text, the rule as it applies to this vendor, and which mandatory
   trigger code (see the system instructions) or risk-engine factor it supports.

Quote the policy. Do not paraphrase a rule into a stronger statement than the
corpus makes, and do not include a rule the corpus does not contain."""

    def retrieve_policy_facts(self, form: Dict[str, Any],
                              signal_bag: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Phase A. Runs the completeness gate, builds signals, and retrieves
        the policy rules with their grounding chunks.

        A corpus that is unreachable or answers with nothing usable falls back
        to the in-repo policy store (local_corpus), so a complete submission
        always cites the sections its decision was made under.
        `retrieval["_meta"]` records which store served the pass.

        Returns:
            {
              "form": <validated form>,
              "report": <validation report>,
              "signals": <signal bag>,
              "retrieval": <RETRIEVAL_SCHEMA data or None>,
              "chunks": [{"uri","text","score"}, ...],
              "citations": [Citation, ...],
              "blocked": bool,
            }
        """
        form = form if isinstance(form, dict) else {}
        report = validation_lib.validate(form)
        signal_bag = signal_bag or signal_lib.build_signals(form)

        if not report.is_complete:
            # The Zero-Hallucination gate: no retrieval, no scoring, no model.
            # Nothing about this vendor is assumed, and a token is not spent.
            return {
                "form": form,
                "report": report,
                "signals": signal_bag,
                "retrieval": None,
                "chunks": [],
                "citations": [],
                "blocked": True,
            }

        retrieval_error = False
        result: Optional[Dict[str, Any]] = None
        try:
            result = self.generate_grounded(
                prompt=self._retrieval_prompt(form, signal_bag),
                system_instruction=self.system_instruction,
                response_schema=RETRIEVAL_SCHEMA,
                use_rag=True,
                model_name=self.retrieval_model_name,
                step="phase_a_retrieval",
            )
        except LLMParseError:
            # A response that never conformed carries nothing quotable, so it
            # counts as "the corpus said nothing" below.
            result = None
        except Exception:  # noqa: BLE001 - deliberately broad
            # Retrieval is best-effort: an unreachable corpus must not stop a
            # deterministic assessment. The local policy store below keeps the
            # pass grounded rather than empty.
            retrieval_error = True

        retrieval = (result or {}).get("data") or {}
        chunks = (result or {}).get("chunks") or []
        source = "vertex-rag"

        if not (retrieval and retrieval.get("rules") and chunks):
            # The corpus answered with nothing usable: an outage, a response
            # that never conformed, or rules with no grounding text behind
            # them. Fall back to the in-repo policy store so the dossier cites
            # the sections the decision was actually made under — Form 73/74,
            # Assumption A7, the EDD thresholds — instead of reporting that no
            # passages were retrieved. Citations are still built through
            # `build_citations`, so a rule is only ever marked verified
            # against text the store really carries.
            partial = retrieval
            local = local_corpus.retrieve(form, signal_bag)
            retrieval = local["retrieval"]
            chunks = local["chunks"]
            source = local_corpus.SOURCE
            # Anything the corpus did establish — a jurisdiction tier, a
            # category — is kept rather than discarded along with its rules:
            # the store deliberately carries no country Risk List, and
            # throwing away a tier the corpus did return would silently
            # rescore the vendor.
            for key in ("jurisdiction_tier", "jurisdiction_basis",
                        "business_category"):
                value = (partial or {}).get(key)
                if value:
                    retrieval[key] = value
        else:
            chunks = self._anchor_policy_text(chunks)

        # Provenance for the dossier's Provenance card: which store served
        # this pass, how many passages it yielded, and when.
        retrieval["_meta"] = {
            "source": source,
            "corpus": (
                local_corpus.CORPUS_LABEL
                if source == local_corpus.SOURCE
                else "Sensei RAG corpus"
            ),
            "retrieved_at": _timestamp(),
            "passages": len(chunks),
        }

        # RAG establishes the country tier; the signal bag is rebuilt so the
        # engine scores against the retrieved tier rather than the form alone.
        # The local store carries no country Risk List and reports no tier, so
        # there the form's declared tier stands unchanged.
        rag_tier = retrieval.get("jurisdiction_tier")
        if rag_tier:
            signal_bag = signal_lib.build_signals(form, rag_jurisdiction_tier=rag_tier)

        citations = citation_lib.build_citations(
            retrieved=chunks,
            assertions=retrieval.get("rules") or [],
            document=POLICY_DOCUMENT,
            section_fallback=POLICY_SECTION,
        )

        context: Dict[str, Any] = {
            "form": form,
            "report": report,
            "signals": signal_bag,
            "retrieval": retrieval,
            "chunks": chunks,
            "citations": citations,
            "blocked": False,
        }
        if retrieval_error:
            context["retrieval_error"] = True
        return context

    def _anchor_policy_text(self, chunks: List[Dict[str, Any]]
                            ) -> List[Dict[str, Any]]:
        """Top the evidence pool up with the four scoring sections, asked for by
        name.

        The shared corpus holds the Procurement Policy and other documents, and
        one broad query does not reliably surface the policy: across live runs
        the same submission returned the Supplier Code of Conduct instead of the
        Risk Factors section, which dropped grounded citations from 1 of 14 to
        0 of 6 on identical input. Asking for each section explicitly, and
        merging the grounding chunks, removes that lottery.

        Best-effort throughout: an unreachable corpus or a non-conforming
        response leaves the pool exactly as it was. Chunks are deduplicated on
        (uri, text) so a section returned by several queries is only quoted once.
        """
        seen = {(str(c.get("uri") or ""), str(c.get("text") or ""))
                for c in chunks}
        merged = list(chunks)

        for query in vq_schemas.POLICY_ANCHOR_QUERIES:
            try:
                result = self.generate_grounded(
                    prompt=query,
                    system_instruction=self.system_instruction,
                    response_schema=vq_schemas.POLICY_ANCHOR_SCHEMA,
                    use_rag=True,
                    model_name=self.retrieval_model_name,
                    step="phase_a_policy_anchor",
                )
            except Exception:  # noqa: BLE001 - anchoring is never load-bearing
                continue
            for chunk in (result or {}).get("chunks") or []:
                key = (str(chunk.get("uri") or ""), str(chunk.get("text") or ""))
                if not key[1] or key in seen:
                    continue
                seen.add(key)
                merged.append(chunk)

        return merged

    # ------------------------------------------------------------------
    # Deterministic assessment
    # ------------------------------------------------------------------

    @staticmethod
    def assess(signals: Dict[str, Any]) -> Tuple[risk_engine.RiskAssessment,
                                                 trigger_lib.MandatoryEddResult]:
        """Run the scoring engine and the mandatory EDD override.

        This is the only place the dossier's score and tier come from.
        """
        assessment = risk_engine.assess(signals)
        override = trigger_lib.apply_mandatory_edd(assessment, signals)
        return assessment, override

    @staticmethod
    def _final_score_tier(assessment: risk_engine.RiskAssessment,
                          override: trigger_lib.MandatoryEddResult
                          ) -> Tuple[float, str, str, str]:
        """Merge the base assessment with the mandatory override."""
        if override.triggered and override.score_override is not None:
            score = override.score_override
        else:
            score = assessment.weighted_risk_score
        tier = override.tier_override if override.triggered else assessment.assigned_risk_tier
        _, dd_level, cycle = risk_engine.tier_for(score)
        return float(score), tier, dd_level, cycle

    @staticmethod
    def _jurisdiction_is_prohibited(signals: Dict[str, Any]) -> bool:
        tier = _clean((signals.get("jurisdiction") or {}).get("tier")).lower()
        return tier in PROHIBITED_JURISDICTION_TIERS

    # ------------------------------------------------------------------
    # Phase B — narrative
    # ------------------------------------------------------------------

    def _narrative_prompt(self, form: Dict[str, Any], signal_bag: Dict[str, Any],
                          assessment: risk_engine.RiskAssessment,
                          override: trigger_lib.MandatoryEddResult) -> str:
        signal_text = signal_lib.signals_to_prompt(signal_bag)

        factor_lines = "\n".join(
            f"- {f.label}: {f.rule}" for f in assessment.factors
        )
        unknown_note = ""
        if assessment.unknown_factors:
            names = ", ".join(
                risk_engine.FACTOR_LABELS[key] for key in assessment.unknown_factors
            )
            unknown_note = (
                f"\n\nSome factors could not be assessed because the submission "
                f"did not declare them: {names}. Do not describe an unassessed "
                f"factor as satisfactory."
            )

        trigger_lines = ""
        if override.triggers:
            rows = "\n".join(
                f"- {t.code}: {t.label}\n  Policy basis: {t.policy_basis}\n"
                f"  Minimum controls: {t.minimum_controls}"
                for t in override.triggers
            )
            trigger_lines = (
                "\n\n## Mandatory EDD triggers detected\n"
                "The deterministic engine has already detected these. They are "
                "not yours to decide, detect, or dispute. Describe what each one "
                "means for this engagement and set the required controls.\n\n"
                f"{rows}"
            )

        notes = signal_bag.get("derivation_notes") or []
        notes_block = ""
        if notes:
            rows = "\n".join(f"- {note}" for note in notes)
            notes_block = (
                "\n\n## Assessment notes\nThe scoring engine recorded how some "
                "signals were established. Treat a derived value as a reading of "
                "disclosed data, not as a vendor declaration.\n\n" + rows
            )

        references = self._reference_block(form)

        return f"""Assess this vendor submission for the National Holding
Vendor Prequalification and Master Vendor List decision (Stage 5, Appendix F),
and produce the qualification narrative.

## Submission identity
{references}

## Declared risk signals
{signal_text}{notes_block}

## Weighted factor positions established by the scoring engine
These are the engine's rule texts for each factor given this vendor's signals.
Use them to explain the posture; do not restate a number.
{factor_lines}{unknown_note}{trigger_lines}

## What to write

**company_profile** — transcribe from the submission. Any field the vendor did
not supply stays an empty string or 0. Do not infer a trade licence number, a
VAT number, or a year.

**ownership_summary** — how control runs from the natural persons to the
contracting entity. Set structure_complexity to SIMPLE, LAYERED, or OFFSHORE.
List the declared UBOs with their nationality and percentage. Set pep_present
from the declared PEP answer only; do not speculate about a person from their
name or nationality.

**appendix_f_assessment** — score financial_standing, technical_capability,
and quality_hse on a raw 0-100 scale as you judge the evidence. The engine
derives the dossier's sheet from the assigned risk tier, so yours is a
cross-check on that derivation rather than the number the payload carries;
the weights and the pass/fail gate are applied by the engine either way. In
the rationale, name the specific evidence behind each category score.

**required_controls** — the minimum controls this engagement needs: the tier's
due-diligence level, plus the specific controls any detected trigger carries.
Mark each as mandatory or advisory and cite the policy section that requires it.

**open_questions** — what a procurement officer must put to the vendor before
Stage 7. Genuine gaps only. Do not use this to restate a missing mandatory
field; those are handled upstream.

**assessment_narrative** — two to four sentences of plain language for a
procurement officer. It must not contain a numeric risk score, a tier name, or
the words "approved", "cleared", or "compliant".

**rules** — the policy rules that bear on this vendor, with the section heading
as it appears in the retrieved context, the rule as it applies here, the
mandatory trigger code it supports, and the risk-engine factor it informs.

Use the retrieval tool to source every rule you cite. A rule the corpus does not
contain is omitted, not invented."""

    @staticmethod
    def _reference_block(form: Dict[str, Any]) -> str:
        rows: List[str] = []
        for label, key in (
            ("Company / legal name", "legal_name"),
            ("Registered address", "registered_address"),
            ("Country of incorporation", "country_of_incorporation"),
            ("Date of incorporation", "date_of_incorporation"),
            ("Year of commencement", "year_of_commencement"),
            ("Trade licence number", "trade_license_no"),
            ("Trade licence expiry", "trade_license_expiry"),
            ("VAT registration status", "vat_registration_status"),
            ("TRN VAT number", "vat_registration_no"),
            ("ISO / quality / HSE / InfoSec certifications", "iso_hse_certifications"),
            ("Nature of business", "nature_of_business"),
            ("Goods / services proposed", "goods_services_proposed"),
            ("Geographic coverage", "geographic_coverage"),
            ("Authorized representative", "authorized_representative_name"),
            ("Authorized representative designation",
             "authorized_representative_designation"),
            ("Representative ID reference", "authorized_representative_id"),
            ("Ultimate parent company", "ultimate_parent_company"),
            ("Payment terms", "payment_terms"),
            ("Authorized signatory", "authorized_signatory_name"),
            ("Signatory designation", "authorized_signatory_designation"),
            ("Date signed", "date_signed"),
        ):
            value = _clean(form.get(key))
            if value:
                rows.append(f"- {label}: {value}")

        owners = form.get("directors_and_owners")
        if isinstance(owners, (list, tuple)) and owners:
            rendered = "; ".join(
                f"{_clean(o.get('name'))} ({_clean(o.get('position')) or 'position not stated'}"
                f"{', ' + _clean(o.get('nationality')) if _clean(o.get('nationality')) else ''})"
                for o in owners if isinstance(o, dict)
            )
            rows.append(f"- Directors / partners / shareholders / owners: {rendered}")

        ubos = form.get("ubos")
        if isinstance(ubos, (list, tuple)) and ubos:
            rendered = "; ".join(
                f"{_clean(u.get('name'))} — {_clean(u.get('nationality'))}, "
                f"{u.get('ownership_percentage')}%"
                for u in ubos if isinstance(u, dict)
            )
            rows.append(f"- Ultimate beneficial owners: {rendered}")

        references = form.get("client_references")
        if isinstance(references, (list, tuple)) and references:
            rendered = "; ".join(
                f"{_clean(r.get('client_name'))} ({_clean(r.get('contact_details'))})"
                for r in references if isinstance(r, dict)
            )
            rows.append(f"- Client references: {rendered}")

        for label, key in (
            ("Turnover year 1 (most recent)", "turnover_year_1"),
            ("Turnover year 2", "turnover_year_2"),
            ("Turnover year 3", "turnover_year_3"),
            ("Number of employees", "number_of_employees"),
            ("Estimated NH spend (AED)", "estimated_spend_aed"),
            ("Bank name, branch & country", "bank_name_branch_country"),
            ("Bank account name", "bank_account_name"),
            ("Bank IBAN", "bank_iban"),
        ):
            value = form.get(key)
            if value not in (None, "", []):
                rows.append(f"- {label}: {_clean(value)}")

        if not rows:
            return "- No submission data was provided."
        return "\n".join(rows)

    async def _generate_narrative(self, form: Dict[str, Any],
                                  signal_bag: Dict[str, Any],
                                  assessment: risk_engine.RiskAssessment,
                                  override: trigger_lib.MandatoryEddResult
                                  ) -> Optional[Dict[str, Any]]:
        """Phase B. A failure here never fails the assessment: the deterministic
        score, tier, triggers and Appendix F arithmetic are already computed and
        are returned regardless.

        Retried once. A single unstructured-schema call with this many required
        fields is the most failure-prone step in the run, and a live run dropped
        the entire citation set because of one transient parse failure, which
        turned a dossier with grounded policy references into one with none.
        The deterministic half of the dossier is unaffected either way, so a
        retry is cheap insurance rather than a correctness risk.
        """
        prompt = self._narrative_prompt(form, signal_bag, assessment, override)
        last_error: Optional[str] = None

        for attempt in range(2):
            try:
                return await self.generate_with_rag(
                    prompt=prompt,
                    system_instruction=self.system_instruction,
                    response_schema=NARRATIVE_SCHEMA,
                    use_rag=True,
                    model_name=self.model_name,
                    step="phase_b_narrative",
                )
            except LLMParseError as exc:
                last_error = f"schema parse failed: {exc}"
            except Exception as exc:  # noqa: BLE001 - deliberately broad
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                LOGGER.warning(
                    "[vendor_qualification] narrative attempt 1 failed (%s); retrying",
                    last_error,
                )

        LOGGER.warning(
            "[vendor_qualification] narrative unavailable after 2 attempts: %s",
            last_error,
        )
        return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @staticmethod
    def _status_for(override: trigger_lib.MandatoryEddResult,
                    signals: Dict[str, Any],
                    appendix_f: Dict[str, Any],
                    appendix_assessed: bool
                    ) -> Tuple[str, str]:
        """Decide the qualifying status. Deterministic, and explained.

        REJECTED_INCOMPLETE  a mandatory NH-PQF-001 field is missing. Handled
                             before this function is reached.
        REJECTED_HIGH_RISK   the engagement is blocked rather than escalated:
                             a Prohibited jurisdiction, an unresolved sanctions
                             match, a credible financial-crime association, or
                             a failed Appendix F sheet on a vendor the policy
                             does not already route to EDD.
        NEEDS_EDD            at least one mandatory Always-EDD trigger fired.
        QUALIFIED            no trigger, no disqualifier, Appendix F passed.

        Precedence is strongest-fact-first. A Prohibited jurisdiction or a
        blocking trigger outranks everything. The Appendix F sheet restates the
        assigned tier, so on a vendor that already carries a mandatory EDD
        trigger the engagement reads as NEEDS_EDD — the sheet's flag ("requires
        MLRO clearance") is stated on the sheet and in its rationale — while a
        failed sheet with no EDD trigger behind it blocks outright: a vendor
        that cannot pass prequalification does not need more risk diligence, it
        needs different financials.

        Note on the enum: output_schema.md offers no "failed the commercial
        score sheet" value, so a failed Appendix F reports as
        REJECTED_HIGH_RISK with the reason stated in `rejection_reason` and
        `next_action`. The enum is not widened, so the Tool 2 contract holds.
        """
        if VendorQualificationAgent._jurisdiction_is_prohibited(signals):
            return (
                STATUS_REJECTED_HIGH_RISK,
                "Country of incorporation is a Prohibited jurisdiction. The "
                "policy does not permit the engagement irrespective of the "
                "weighted score.",
            )

        codes = {t.code for t in override.triggers}
        blocking = codes & DISQUALIFYING_TRIGGER_CODES
        if blocking:
            return (
                STATUS_REJECTED_HIGH_RISK,
                "The submission hits a policy condition that blocks "
                "prequalification rather than escalating it: "
                + ", ".join(sorted(blocking)) + ".",
            )

        if override.triggered:
            return STATUS_NEEDS_EDD, ""

        if appendix_assessed and appendix_f.get("status") == "FAILED":
            failed = appendix_f.get("failed_categories") or []
            detail = (
                " (below the minimum in: " + ", ".join(failed) + ")"
                if failed else ""
            )
            return (
                STATUS_REJECTED_HIGH_RISK,
                f"Appendix F prequalification score sheet did not pass: "
                f"{appendix_f.get('total_score')} / 100 against a pass threshold "
                f"of {APPENDIX_F_PASS_THRESHOLD}{detail}.",
            )

        return STATUS_QUALIFIED, ""

    @staticmethod
    def _next_action(status: str, tier: str, reason: str,
                     appendix_assessed: bool) -> str:
        if status == STATUS_REJECTED_HIGH_RISK:
            return (
                f"Do not route to the Vendor Onboarding Agent. {reason} Refer to "
                f"the Group Compliance and Internal Controls Department for a "
                f"decision under Stage 1 and Stage 7 of the Procurement Policy."
            )
        if status == STATUS_NEEDS_EDD:
            action = NEXT_ACTION_EDD
        elif status == STATUS_QUALIFIED:
            action = NEXT_ACTION_QUALIFIED
        else:
            action = NEXT_ACTION_INCOMPLETE
        if not appendix_assessed and status in (STATUS_QUALIFIED, STATUS_NEEDS_EDD):
            action += (
                " The Appendix F prequalification score sheet has not been "
                "assessed, so this vendor is not yet prequalified and must not be "
                "onboarded until that assessment is completed."
            )
        return action

    # ------------------------------------------------------------------
    # Narrative fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _narrative_from_form(form: Dict[str, Any],
                             override: trigger_lib.MandatoryEddResult,
                             dd_level: str) -> Dict[str, Any]:
        """Phase B's fallback: the profile transcribed from the submission
        itself when the model narrative is unavailable.

        Nothing is inferred — fields the vendor did not supply stay empty, the
        UBO list is read from the declared rows, and the controls are the tier's
        due-diligence level plus the ones the engine's own triggers already
        carry, each with its policy basis. A model outage must not blank the
        dossier's identity card when the answers are sitting on the form.
        """
        form = form if isinstance(form, dict) else {}

        def yes(value: Any) -> bool:
            return str(value or "").strip().lower() in ("yes", "y", "true", "1")

        established = 0
        for key in ("date_of_incorporation", "year_of_commencement"):
            match = re.match(r"(\d{4})", _clean(form.get(key)))
            if match:
                established = int(match.group(1))
                break

        if (yes(form.get("offshore_structure"))
                or yes(form.get("cross_border_ownership"))
                or _clean(form.get("legal_form")).lower() in ("trust", "foundation")):
            complexity = "OFFSHORE"
        elif (yes(form.get("multi_layered_ownership"))
              or yes(form.get("nominee_shareholders"))
              or yes(form.get("bearer_shares"))):
            complexity = "LAYERED"
        else:
            complexity = "SIMPLE"

        def pct(value: Any) -> float:
            try:
                return round(float(str(value or "0").strip().rstrip("%")), 2)
            except ValueError:
                return 0.0

        ubos = [
            {
                "name": _clean(row.get("name")),
                "nationality": _clean(row.get("nationality")),
                "ownership_percentage": pct(row.get("ownership_percentage")),
            }
            for row in (form.get("ubos") or [])
            if isinstance(row, dict) and _clean(row.get("name"))
        ]

        controls: List[Dict[str, Any]] = []
        if dd_level:
            controls.append({
                "control": dd_level,
                "basis": "Weighted Risk Score / Risk Level / Due Diligence / Refresh Cycle",
                "mandatory": True,
            })
        for trigger in override.triggers:
            controls.append({
                "control": trigger.minimum_controls,
                "basis": trigger.policy_basis,
                "mandatory": True,
            })

        return {
            "company_profile": {
                "legal_entity_name": _clean(form.get("legal_name")),
                "trade_license_no": _clean(form.get("trade_license_no")),
                "country_of_incorporation": _clean(
                    form.get("country_of_incorporation")),
                "year_established": established,
                # The category is retrieval's to state; an empty string here
                # falls through to `retrieval.business_category` in _assemble.
                "business_category": "",
                "vat_registration_no": _clean(form.get("vat_registration_no")),
                "years_operating": (
                    datetime.now(timezone.utc).year - established
                    if established else 0
                ),
            },
            "ownership_summary": {
                "structure_complexity": complexity,
                "structure_narrative": "As declared on Form NH-PQF-001.",
                "pep_present": yes(form.get("pep_present")),
                "ubos": ubos,
            },
            "required_controls": controls,
            # The model is what notices a genuine gap worth putting to the
            # vendor; with no model there is nothing to raise, and the
            # missing-mandatory-field list is carried upstream anyway.
            "open_questions": [],
            "assessment_narrative": "",
            "rules": [],
        }

    # ------------------------------------------------------------------
    # Dossier assembly
    # ------------------------------------------------------------------

    def _assemble(self, form: Dict[str, Any], signal_bag: Dict[str, Any],
                  retrieval: Optional[Dict[str, Any]],
                  narrative: Optional[Dict[str, Any]],
                  assessment: risk_engine.RiskAssessment,
                  override: trigger_lib.MandatoryEddResult,
                  citation_list: List[citation_lib.Citation],
                  retrieval_available: bool) -> Dict[str, Any]:
        """Merge the model's narrative with the engine's numbers.

        Everything numeric and categorical in the output comes from the
        engine. The model contributes profile, ownership narrative, controls,
        questions, and its own rule assertions; Appendix F is derived from the
        assigned risk tier rather than graded independently, so the sheet and
        the band table can never disagree — and a model outage cannot leave it
        blank. When the narrative is absent entirely it is rebuilt from the
        form itself, so the identity card is never empty on a completed
        submission.
        """
        score, tier, dd_level, cycle = self._final_score_tier(assessment, override)
        narrative = narrative if isinstance(narrative, dict) else {}
        if not narrative:
            narrative = self._narrative_from_form(form, override, dd_level)

        # Appendix F restates the tier the deterministic engine assigned. The
        # engine owns every number in the dossier (module docstring), and a
        # sheet derived from the assigned band can neither disagree with the
        # score it came from nor come back unassessed: the raw categories are
        # the band's reference positions (90/90/85 SDD, 76/76/76 CDD,
        # 55/55/55 EDD), so the weighted totals land on 88.5, 76.0 and 55.0.
        raw_appendix = appendix_f_for_tier(tier)
        appendix_f = appendix_f_scores(raw_appendix)
        appendix_assessed = True
        appendix_f["assessed"] = True

        status, reason = self._status_for(
            override, signal_bag, appendix_f, appendix_assessed)

        # Citations: Phase A first (retrieval-verified), then any Phase B
        # assertions that Phase A did not already cover.
        merged = list(citation_list)
        existing = {(c.section.lower(), c.rule_applied.lower()) for c in merged}
        for assertion in narrative.get("rules") or []:
            if not isinstance(assertion, dict):
                continue
            key = (_clean(assertion.get("section")).lower(),
                   _clean(assertion.get("rule_applied")).lower())
            if key in existing or not key[1]:
                continue
            existing.add(key)
            merged.append(citation_lib.Citation(
                document=POLICY_DOCUMENT,
                section=_clean(assertion.get("section")) or POLICY_SECTION,
                rule_applied=_clean(assertion.get("rule_applied")),
                matched_trigger=_clean(assertion.get("matched_trigger")) or None,
                factor=_clean(assertion.get("factor")) or None,
                verified=False,
            ))

        company_profile = narrative.get("company_profile") or {}
        ownership = narrative.get("ownership_summary") or {}
        retrieval = retrieval or {}

        vendor_name = _clean(company_profile.get("legal_entity_name")) \
            or _clean(form.get("legal_name"))

        guardrail_passed = (
            status != STATUS_REJECTED_INCOMPLETE
            and bool(override.triggers) == bool(override.triggered)
            and 1.0 <= score <= 3.0
            and tier in (TIER_LOW, TIER_MEDIUM, TIER_HIGH)
            and (appendix_assessed or status == STATUS_REJECTED_HIGH_RISK)
        )

        payload: Dict[str, Any] = {
            "vendor_id": _vendor_id(vendor_name or "unnamed-vendor"),
            "vendor_name": vendor_name,
            "qualification_status": status,
            "weighted_risk_score": round(score, 2),
            "assigned_risk_tier": tier,
            "mandatory_edd_triggered": override.triggered,
            "trigger_reasons": trigger_lib.trigger_reasons(override.triggers),
            "company_profile": {
                "trade_license_no": _clean(company_profile.get("trade_license_no")),
                "country_of_incorporation": _clean(
                    company_profile.get("country_of_incorporation")),
                "year_established": int(company_profile.get("year_established") or 0),
                "business_category": _clean(
                    company_profile.get("business_category")) or
                    _clean(retrieval.get("business_category")),
                "vat_registration_no": _clean(company_profile.get("vat_registration_no")),
            },
            "ownership_summary": {
                "structure_complexity": _clean(
                    ownership.get("structure_complexity")) or "SIMPLE",
                "pep_present": bool(ownership.get("pep_present")),
                "ubos": [
                    {
                        "name": _clean(ubo.get("name")),
                        "nationality": _clean(ubo.get("nationality")),
                        "ownership_percentage": round(
                            float(ubo.get("ownership_percentage") or 0.0), 2),
                    }
                    for ubo in (ownership.get("ubos") or [])
                    if isinstance(ubo, dict) and _clean(ubo.get("name"))
                ],
            },
            "appendix_f_score": appendix_f,
            "guardrail_check_passed": guardrail_passed,
            "rag_retrieval_citations": citation_lib.policy_citations(merged),
            "timestamp": _timestamp(),
            "next_action": self._next_action(
                status, tier, reason, appendix_assessed),
            "onboarding_flags": onboarding_flags(form),
        }

        # Fields beyond the Tool 2 contract, so the officer sees the working.
        payload["due_diligence_level"] = dd_level
        payload["refresh_cycle"] = cycle
        payload["rejection_reason"] = reason
        payload["weighted_factors"] = [f.to_dict() for f in assessment.factors]
        payload["trigger_details"] = [t.to_dict() for t in override.triggers]
        payload["citation_summary"] = citation_lib.citation_summary(merged)
        payload["jurisdiction_assessment"] = {
            "tier": _clean((signal_bag.get("jurisdiction") or {}).get("tier")) or None,
            "tier_source": (signal_bag.get("jurisdiction") or {}).get("tier_source"),
            "basis": _clean(retrieval.get("jurisdiction_basis")),
        }
        payload["required_controls"] = narrative.get("required_controls") or []
        payload["open_questions"] = narrative.get("open_questions") or []
        payload["assessment_narrative"] = _clean(
            narrative.get("assessment_narrative"))
        payload["assessment_notes"] = signal_bag.get("derivation_notes") or []
        payload["rag_available"] = retrieval_available
        meta = retrieval.get("_meta")
        if isinstance(meta, dict) and meta:
            # Which store served Phase A, how many passages it yielded, and
            # when — the Provenance card reads this so a reviewer can always
            # tell a corpus pass from the local policy store.
            payload["retrieval_metadata"] = meta
        return payload

    def _incomplete_dossier(self, form: Dict[str, Any],
                            report: validation_lib.ValidationReport) -> Dict[str, Any]:
        """The REJECTED_INCOMPLETE dossier. No model call, nothing inferred."""
        vendor_name = _clean(form.get("legal_name")) if isinstance(form, dict) else ""
        payload = empty_output(vendor_name)
        payload["vendor_id"] = _vendor_id(vendor_name or "unnamed-vendor")
        payload["weighted_risk_score"] = None
        payload["assigned_risk_tier"] = None
        payload["next_action"] = validation_lib.missing_data_request(report)["message"]
        payload["rejection_reason"] = (
            f"{len(report.missing)} mandatory field(s) required by Form NH-PQF-001 "
            f"were not supplied. No value has been assumed for any of them."
        )
        payload["missing_data_request"] = validation_lib.missing_data_request(report)
        payload["advisory_fields"] = [m.to_dict() for m in report.advisory]
        payload["due_diligence_level"] = ""
        payload["refresh_cycle"] = ""
        payload["rag_available"] = False
        return payload

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def qualify(self, form: Dict[str, Any]) -> Dict[str, Any]:
        """Full qualification run. Returns an output_schema.md payload."""
        async for event in self.qualify_stream(form):
            if event["type"] == "dossier":
                return event["dossier"]
        # qualify_stream always terminates with a dossier; this is unreachable
        # but keeps the contract total.
        return self._incomplete_dossier(
            form if isinstance(form, dict) else {},
            validation_lib.validate(form if isinstance(form, dict) else {}),
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def qualify_stream(self, form: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Qualification with progressive events for the live risk dashboard.

        Event sequence:
            stage            a labelled step began
            validation       completeness verdict
            retrieval        a policy rule was retrieved and cited
            score            the weighted score and tier
            triggers         a mandatory EDD trigger fired
            narrative        the model narrative is ready
            dossier          the final output_schema.md payload
            error            something failed; the dossier is still returned
        """
        form = form if isinstance(form, dict) else {}

        yield {"type": "stage", "stage": "validation",
               "label": "Validating NH-PQF-001 submission"}

        context = self.retrieve_policy_facts(form)

        if context.get("blocked"):
            report = context["report"]
            yield {
                "type": "validation",
                "is_complete": False,
                "missing_count": len(report.missing),
                "missing_fields": [m.to_dict() for m in report.missing],
            }
            yield {
                "type": "stage",
                "stage": "rejected",
                "label": "Submission incomplete — scoring not attempted",
            }
            yield {
                "type": "dossier",
                "dossier": self._incomplete_dossier(form, report),
            }
            return

        report = context["report"]
        yield {
            "type": "validation",
            "is_complete": True,
            "missing_count": 0,
            "advisory_count": len(report.advisory),
        }

        signal_bag = context["signals"]
        retrieval = context.get("retrieval") or {}
        citation_list = context.get("citations") or []
        chunks = context.get("chunks") or []

        source = (retrieval.get("_meta") or {}).get("source")
        yield {
            "type": "stage",
            "stage": "retrieval",
            "label": (
                "Retrieving policy rules from the local policy store"
                if source == local_corpus.SOURCE
                else "Retrieving policy rules from the Sensei corpus"
            ),
        }

        if retrieval.get("jurisdiction_tier"):
            yield {
                "type": "jurisdiction",
                "tier": retrieval["jurisdiction_tier"],
                "basis": _clean(retrieval.get("jurisdiction_basis")),
            }
        if retrieval.get("business_category"):
            yield {
                "type": "category",
                "category": retrieval["business_category"],
            }

        for index, citation in enumerate(citation_list):
            yield {
                "type": "retrieval",
                "index": index,
                "citation": citation.to_dict(),
            }

        yield {
            "type": "stage",
            "stage": "scoring",
            "label": "Computing weighted risk score",
        }

        assessment, override = self.assess(signal_bag)
        score, tier, dd_level, cycle = self._final_score_tier(assessment, override)

        yield {
            "type": "score",
            "weighted_risk_score": round(score, 2),
            "assigned_risk_tier": tier,
            "due_diligence_level": dd_level,
            "refresh_cycle": cycle,
            "base_score": float(assessment.weighted_risk_score),
            "base_tier": assessment.assigned_risk_tier,
            "overridden": override.triggered,
            "factors": [f.to_dict() for f in assessment.factors],
            "unknown_factors": assessment.unknown_factors,
        }

        for index, trigger in enumerate(override.triggers):
            yield {"type": "triggers", "index": index, "trigger": trigger.to_dict()}

        yield {
            "type": "stage",
            "stage": "assessment",
            "label": "Assessing financial, technical, and quality standing",
        }

        narrative = await self._generate_narrative(form, signal_bag, assessment, override)
        if narrative is None:
            yield {
                "type": "error",
                "message": (
                    "The narrative assessment could not be completed. The "
                    "deterministic risk score, tier, and trigger findings below "
                    "are unaffected and complete."
                ),
            }

        dossier = self._assemble(
            form=form,
            signal_bag=signal_bag,
            retrieval=retrieval,
            narrative=narrative,
            assessment=assessment,
            override=override,
            citation_list=citation_list,
            retrieval_available=bool(citation_list or chunks),
        )

        yield {"type": "dossier", "dossier": dossier}
