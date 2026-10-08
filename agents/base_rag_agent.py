"""
Base RAG Agent for the Vendor Qualification & Risk Tiering workflow (Tool 1).

Follows the SENSEI pattern established by the Supplier Contract Review
workflow:
- Vertex AI RAG Engine retrieval bound as a tool on the model
- Structured JSON output enforced with a schema gate
- Immune JSON parsing (json5 -> dirtyjson -> LLM repair)
- Consistent error handling

Two additions specific to compliance work:

1. `generate_grounded` runs a retrieval-first pass and returns the grounding
   chunks alongside the parsed object, so citations can be backed by the
   verbatim source text instead of the model's own description of it.

2. Usage events are tagged feature="vendor_qualification" and
   event_type="workflow_vendor_qualification_agent_call", kept separate from
   the contract-review events in the shared admin_events collection.

The RAG corpus is shared with the Supplier Contract Review workflow
(rag-data-full-corpus), which already contains the National Holding
Procurement Policy. Nothing here is specific to a single corpus, so pointing
RAG_CORPUS at a dedicated corpus later is a one-variable change.
"""

import os
import json
import json5
import dirtyjson
import regex as re
from typing import Any, Dict, List, Optional
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool, GenerationConfig
import vertexai

from citations import grounding_chunks


class LLMParseError(Exception):
    """Raised when JSON parsing fails after all attempts."""
    pass


# Vertex rejects the `additionalProperties` keyword, so the schema is stripped
# before it is sent and key-whitelisting is enforced locally by
# `_conforms_to_schema` instead.
_ADDITIONAL_PROPERTIES = "additionalProperties"

EVENT_TYPE = "workflow_vendor_qualification_agent_call"
FEATURE = "vendor_qualification"
ANONYMOUS = "workflow_anonymous"


class BaseRAGAgent:
    """
    Base class for all RAG-enabled agents.

    Usage:
        class VendorQualificationAgent(BaseRAGAgent):
            def __init__(self):
                super().__init__(model_name="gemini-2.5-pro")

            async def qualify(self, ...):
                return await self.generate_with_rag(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    response_schema=QUALIFICATION_NARRATIVE_SCHEMA,
                    use_rag=True,
                )
    """

    def __init__(self, model_name: str = "gemini-2.5-pro",
                 similarity_top_k: int = 15):
        """
        Initialize base agent.

        Args:
            model_name: "gemini-2.5-pro" (policy reasoning) or
                        "gemini-2.5-flash" (retrieval pass, cheaper/faster)
            similarity_top_k: chunks retrieved per RAG query. 15 matches the
                        contract-review workflow and is enough to span the
                        Risk Scoring Matrix, the Mandatory High-Risk
                        Classification list, and the Vendor Category matrix.
        """
        self.model_name = model_name
        self.similarity_top_k = similarity_top_k
        # The model actually used for the current call. Phase A runs on flash
        # while self.model_name is pro, so usage must be logged against the
        # model that served the request, not the configured default.
        self._active_model_name = model_name
        self.project_id = os.environ.get("PROJECT_ID", "test-rag-corpus-project")
        self.location = os.environ.get("LOCATION", "us-west1")
        self.rag_corpus = os.environ.get(
            "RAG_CORPUS",
            "projects/test-rag-corpus-project/locations/us-west1/ragCorpora/137359788634800128"
        )

        # Initialize Vertex AI once per agent instance
        vertexai.init(project=self.project_id, location=self.location)

    # ------------------------------------------------------------------
    # Model + retrieval
    # ------------------------------------------------------------------

    def get_rag_tool(self, similarity_top_k: Optional[int] = None) -> Tool:
        """Create RAG retrieval tool."""
        return Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=[rag.RagResource(rag_corpus=self.rag_corpus)],
                    similarity_top_k=similarity_top_k or self.similarity_top_k,
                )
            )
        )

    def get_model(self, system_instruction: str, use_rag: bool = True,
                  model_name: Optional[str] = None) -> GenerativeModel:
        """Create Gemini model with optional RAG retrieval."""
        tools = [self.get_rag_tool()] if use_rag else []
        return GenerativeModel(
            model_name=model_name or self.model_name,
            tools=tools,
            system_instruction=system_instruction,
        )

    # ------------------------------------------------------------------
    # JSON recovery
    # ------------------------------------------------------------------

    def fix_json_with_llm(self, broken_json: str) -> str:
        """Use LLM to fix malformed JSON."""
        fix_model = GenerativeModel(model_name="gemini-2.5-pro")
        prompt = f"Fix this JSON syntax and return only valid JSON:\n\n{broken_json}"
        response = fix_model.generate_content(prompt)
        fixed_text = response.text.strip()
        fixed_text = fixed_text.replace('```json', '').replace('```', '')
        return fixed_text.strip()

    def strip_markdown_fences(self, text: str) -> str:
        """Remove markdown code fences from text."""
        return re.sub(
            r"```(?:json|markdown)?\s*([\s\S]*?)```",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )

    def extract_json_candidate(self, text: str) -> str:
        """Extract the largest balanced JSON object from text."""
        matches = re.findall(
            r"""
            \{
                (?:
                    [^{}]++
                    |
                    (?R)
                )*
            \}
            """,
            text,
            flags=re.VERBOSE,
        )
        return max(matches, key=len) if matches else text

    def normalize_llm_json(self, text: str) -> str:
        """Normalize JSON by removing trailing commas and comments."""
        t = text.strip()
        t = re.sub(r",(\s*[}\]])", r"\1", t)
        t = re.sub(r"//.*?$", "", t, flags=re.MULTILINE)
        t = re.sub(r"/\*[\s\S]*?\*/", "", t)
        return t

    def safe_parse_llm_json(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON with multiple fallback strategies.

        Attempts:
        1. Standard json.loads
        2. json5.loads (allows trailing commas, comments)
        3. dirtyjson.loads (very permissive)
        4. LLM-based fix

        Raises:
            LLMParseError: If all parsing attempts fail.
        """
        cleaned = self.strip_markdown_fences(text)
        extracted = self.extract_json_candidate(cleaned)
        normalized = self.normalize_llm_json(extracted)

        for parser in (json.loads, json5.loads, dirtyjson.loads):
            try:
                parsed = parser(normalized)
                # Never treat plain prose or a bare string as structured output
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                continue

        try:
            fixed_json = self.fix_json_with_llm(text)
            parsed = json.loads(fixed_json)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

        raise LLMParseError("Failed to recover JSON from model output")

    def _conforms_to_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Recursively validate data against the JSON schema subset used here:
        type / enum / required / items / properties / additionalProperties.
        Extra keys are rejected where additionalProperties is false, so prose
        or junk fields never pass the gate."""
        if not isinstance(schema, dict):
            return True

        enum = schema.get("enum")
        if enum is not None and data not in enum:
            return False

        base = schema.get("type")
        if base == "array":
            items = schema.get("items")
            if not isinstance(data, list):
                return False
            if items is not None:
                return all(self._conforms_to_schema(item, items) for item in data)
            return True
        if base == "object":
            if not isinstance(data, dict):
                return False
            props = schema.get("properties", {})
            required = schema.get("required", [])
            if any(k not in data for k in required):
                return False
            if schema.get(_ADDITIONAL_PROPERTIES) is False:
                if any(k not in props for k in data):
                    return False
            return all(
                k in props and self._conforms_to_schema(v, props[k])
                for k, v in data.items()
            )
        if base == "string":
            return isinstance(data, str)
        if base == "integer":
            return isinstance(data, bool) is False and isinstance(data, int)
        if base == "number":
            return isinstance(data, bool) is False and isinstance(data, (int, float))
        if base == "boolean":
            return isinstance(data, bool)
        if base == "null":
            return data is None
        if schema.get("anyOf"):
            return any(self._conforms_to_schema(data, opt) for opt in schema["anyOf"])
        return True

    @staticmethod
    def _strip_additional_properties(schema: Any) -> Any:
        """Vertex's protobuf Schema has no `additionalProperties` field and
        raises ParseError if it is present."""
        if isinstance(schema, list):
            return [BaseRAGAgent._strip_additional_properties(s) for s in schema]
        if isinstance(schema, dict):
            return {
                k: BaseRAGAgent._strip_additional_properties(v)
                for k, v in schema.items()
                if k != _ADDITIONAL_PROPERTIES
            }
        return schema

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate_with_rag(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: Dict[str, Any],
        use_rag: bool = True,
        model_name: Optional[str] = None,
        step: str = "generate_with_rag",
    ) -> Dict[str, Any]:
        """
        Generate structured output with optional RAG retrieval.

        Passes `response_schema` as a strict JSON schema constraint through
        GenerationConfig so Gemini is bound to the expected shape. Parsed
        output is validated against that schema and regenerated once if it
        drifts; only schema-conforming dicts are returned to the caller.
        """
        self._active_model_name = model_name or self.model_name
        model = self.get_model(system_instruction, use_rag=use_rag,
                               model_name=model_name)

        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=self._strip_additional_properties(response_schema),
        )

        retry_note = (
            "\n\nYour previous output did not conform to the required JSON "
            "schema. Respond with ONLY a single JSON object exactly matching "
            "the schema — no prose, no markdown fences, no duplicated source "
            "text."
        )

        for attempt in (1, 2):
            response = model.generate_content(prompt,
                                              generation_config=generation_config)

            # Usage tracking — best-effort, never breaks generation
            try:
                self._log_usage(response, step)
            except Exception:
                pass

            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                try:
                    result = self.safe_parse_llm_json(response.text)
                except LLMParseError:
                    result = None

            if isinstance(result, dict) and self._conforms_to_schema(result,
                                                                     response_schema):
                return result

            if attempt == 1:
                prompt = prompt + retry_note
                continue

        raise LLMParseError(
            "Model output did not conform to the required JSON schema after retry"
        )

    def generate_grounded(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: Dict[str, Any],
        use_rag: bool = True,
        model_name: Optional[str] = None,
        step: str = "generate_grounded",
        max_attempts: int = 2,
    ) -> Dict[str, Any]:
        """
        Structured generation that also returns the retrieved source chunks.

        Returns:
            {"data": <schema-conforming dict>,
             "chunks": [{"uri", "text", "score"}, ...]}

        The chunks are the retrieved corpus text that was actually placed in
        the model's context. callers.citations.build_citations() uses them to
        attach a verbatim quote and a source file to every citation, so a
        citation never rests on the model's own description of a source.

        This is the Phase A retrieval pass for the qualification workflow; it
        is also the fallback for Phase B when the narrative pass returns
        nothing schema-conforming.
        """
        self._active_model_name = model_name or self.model_name
        model = self.get_model(system_instruction, use_rag=use_rag,
                               model_name=model_name)
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=self._strip_additional_properties(response_schema),
        )
        retry_note = (
            "\n\nYour previous output did not conform to the required JSON "
            "schema. Respond with ONLY a single JSON object exactly matching "
            "the schema — no prose, no markdown fences."
        )

        collected: List[Dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            current_prompt = prompt if attempt == 1 else prompt + retry_note
            response = model.generate_content(current_prompt,
                                              generation_config=generation_config)
            try:
                self._log_usage(response, step)
            except Exception:
                pass

            for chunk in grounding_chunks(response):
                collected.append(chunk)

            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                try:
                    result = self.safe_parse_llm_json(response.text)
                except LLMParseError:
                    result = None

            if isinstance(result, dict) and self._conforms_to_schema(result,
                                                                     response_schema):
                return {"data": result, "chunks": _dedupe_chunks(collected)}

        return {"data": None, "chunks": _dedupe_chunks(collected)}

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    def _log_usage(self, response, step: str) -> None:
        try:
            from admin_logger import log_token_usage
            usage = self.get_token_usage(response)
            if not usage or not usage.get("total_tokens"):
                return
            log_token_usage(
                EVENT_TYPE,
                ANONYMOUS,
                usage,
                {
                    "provider": "vertex",
                    "model": self._active_model_name or self.model_name,
                    "agent": self.__class__.__name__,
                    "feature": FEATURE,
                    "step": step,
                },
            )
        except Exception:
            pass

    def get_token_usage(self, response) -> Dict[str, int]:
        """Extract token usage from a response."""
        if hasattr(response, 'usage_metadata'):
            return {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop repeated chunks. Vertex returns overlapping 1024-token windows, so
    the same paragraph commonly appears two or three times."""
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        if not text:
            continue
        key = text[:220]
        if key in seen:
            continue
        seen.add(key)
        unique.append({**chunk, "text": text})
    unique.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
    return unique
