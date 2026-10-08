"""
Text extraction for smart auto-fill — Tool 1.

The only job here is turning an uploaded Trade Licence or Company Profile into
text. Field mapping lives in `company_profile`, which is the part that actually
decides what a value means.

Formats
-------
PDF (text layer) and .docx. Both are what a supplier or their notary actually
sends, and a notary template is very often Word. Legacy .doc is *not* supported:
it is a different binary format, `python-docx` cannot read it, and no browser
can upload one that renders correctly anyway. `.doc` is refused with a message
saying to save as .docx, which is a one-click fix for the person sending it.

Scope, and why OCR is not here
------------------------------
This reads text that the file already contains, and nothing else. A scanned or
photographed licence has no text layer, so `extract_text` returns
`extracted=False` with a reason naming that possibility, and the wizard falls
back to manual entry.

That is a real limitation rather than an oversight. OCR needs RapidOCR plus an
ONNX runtime and a model download — several hundred megabytes and a cold start —
and Tool 1 has no other need for it, because the dossier attachments are read
by an officer, not parsed. Importing Tool 2's extraction stack was the other
option and was rejected for the same reason it was not already shared: each
backend here deploys independently, and pulling a contract-scanning dependency
into the qualification service to serve a convenience feature is the wrong trade.
A vendor whose licence is a scan types their details, which is exactly what they
would have done before this feature existed.

Everything here is best-effort and side-effect free. `extract_text` never raises:
an unreadable file comes back as a result with a reason, and turning that into a
message is the route's job.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# 50 MB. Raised from 10 MB because issued licences are routinely large: a
# scanned multi-page certificate with an embedded signature image and full
# letterhead runs to tens of megabytes. The parser reads a text layer, so a
# bigger file costs bandwidth and a few hundred ms, not memory — pypdf streams
# and python-docx reads the one XML part it needs.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# The extension is a hint; the file's magic bytes are what is actually trusted.
ALLOWED_UPLOAD_EXTENSIONS = (".pdf", ".docx")

# Legacy binary .doc. Named separately so it can be refused with a useful
# message instead of the generic "not a PDF".
LEGACY_DOC_EXTENSION = ".doc"

# Below this many characters a document's text counts as absent. A one-page
# licence is a few hundred characters, so a scanned page lands well under it.
_MIN_TEXT_LAYER_CHARS = 40

_PDF_MAGIC = b"%PDF-"
# .docx is a zip archive. So is .xlsx, .pptx and .jar, so the magic bytes only
# get us as far as "a zip"; `_docx_text` then insists on `word/document.xml`
# being inside it.
_ZIP_MAGIC = b"PK\x03\x04"


class UnsupportedUpload(Exception):
    """The upload is not a PDF or a .docx."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UploadTooLarge(Exception):
    """The upload exceeds MAX_UPLOAD_BYTES."""

    def __init__(self, size: int) -> None:
        super().__init__(
            f"File is {size:,} bytes; the limit is {MAX_UPLOAD_BYTES:,} bytes (50 MB)."
        )
        self.size = size


@dataclass
class ExtractionResult:
    """What came out of the upload, or why nothing did."""

    extracted: bool
    text: str = ""
    pages: int = 0
    chars: int = 0
    reason: Optional[str] = None
    #: Per-page notes, e.g. one encrypted page among twenty readable ones.
    warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extracted": self.extracted,
            # The text is not returned to the browser. It is a licence or a bank
            # letter, and the officer has just seen the field values that matter.
            "chars": self.chars,
            "pages": self.pages,
            "warnings": list(self.warnings),
        }


def _extension(name: str) -> str:
    dot = (name or "").rfind(".")
    return name[dot:].lower() if dot >= 0 else ""


def _looks_like_pdf(data: bytes) -> bool:
    return data[: len(_PDF_MAGIC)] == _PDF_MAGIC


def _looks_like_zip(data: bytes) -> bool:
    return data[: len(_ZIP_MAGIC)] == _ZIP_MAGIC


def _docx_text(data: bytes) -> str:
    """Flatten a .docx into label-per-line text.

    A .docx is a zip of XML parts, and the text lives in `word/document.xml`.
    What matters here is *line shape*, because that is what `company_profile`
    matches on: it recognises `Label: value` and `Label value`. Word therefore
    has to be flattened in a way that produces those shapes rather than one
    enormous paragraph.

    Two things this handles that a naive `"\n".join(paragraph.text)` does not:

      - Tables. An issued licence or a company profile built from a notary
        template is very often a two-column table of labels and values. Each
        row is emitted as `Label value`, which the mapper reads.
      - Headers and footers. Issuers, licence numbers and dates often live
        there and nowhere else in the body.
    """
    from docx import Document
    import io

    document = Document(io.BytesIO(data))
    lines: list = []

    def add(text: str) -> None:
        text = (text or "").strip()
        if text:
            lines.append(text)

    # Letterhead first: an issuer or company name in the header is document
    # order, and `company_profile` takes the first confident match per field.
    for section in document.sections:
        for part in (section.header, section.footer):
            for paragraph in part.paragraphs:
                add(paragraph.text)
            for table in part.tables:
                _add_table(table, add)

    for paragraph in document.paragraphs:
        add(paragraph.text)

    for table in document.tables:
        _add_table(table, add)

    return "\n".join(lines)


def _add_table(table, add) -> None:
    for row in table.rows:
        cells: list = []
        # A merged cell is reported once per grid column it spans, so the same
        # text arrives several times. Collapse repeats before building the row,
        # or a merged header becomes "Label Label Label".
        for cell in row.cells:
            text = (cell.text or "").strip()
            if text and (not cells or cells[-1] != text):
                cells.append(text)
        if not cells:
            continue
        if len(cells) == 1:
            add(cells[0])
        else:
            # "Label value" — the space-separated form the mapper also reads.
            # The colon is deliberately absent: it is not in the source file.
            add(" ".join(cells))


def _pdf_text(data: bytes) -> ExtractionResult:
    try:
        from pypdf import PdfReader
        import io
    except ImportError:  # pragma: no cover - deployment error, not input
        return ExtractionResult(
            extracted=False,
            reason="PDF text extraction is not available on this server.",
        )

    try:
        reader = PdfReader(io.BytesIO(data))
        if getattr(reader, "is_encrypted", False):
            # Many "encrypted" PDFs are only permission-locked and still
            # readable once an empty owner password is supplied.
            try:
                if reader.decrypt("") == 0:
                    return ExtractionResult(
                        extracted=False,
                        reason="This PDF is password protected, so its text cannot be read.",
                    )
            except Exception:
                return ExtractionResult(
                    extracted=False,
                    reason="This PDF is password protected, so its text cannot be read.",
                )

        pages = len(reader.pages)
        chunks: list = []
        warnings: list = []
        for index, page in enumerate(reader.pages):
            try:
                chunks.append(page.extract_text() or "")
            except Exception as exc:
                # One unreadable page must not lose the other nineteen.
                chunks.append("")
                warnings.append(f"Page {index + 1} could not be read ({type(exc).__name__}).")
    except Exception as exc:
        return ExtractionResult(
            extracted=False,
            reason=f"This PDF could not be opened ({type(exc).__name__}).",
        )

    text = "\n".join(chunks).strip()
    return _finish(
        text,
        pages=pages,
        warnings=warnings,
        no_text=(
            "No text could be read from this PDF, which usually means it is a "
            "scan or a photograph rather than a digital document. Enter the "
            "details manually, or upload a text-based PDF."
        ),
    )


def _docx_result(data: bytes) -> ExtractionResult:
    try:
        from docx import Document
        import io
    except ImportError:  # pragma: no cover - deployment error, not input
        return ExtractionResult(
            extracted=False,
            reason="Word document extraction is not available on this server.",
        )

    try:
        # A zip that is not a .docx — .xlsx, .pptx, a .jar — has no
        # `word/document.xml`, and python-docx says so less clearly than we can.
        import zipfile

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if "word/document.xml" not in zf.namelist():
                return ExtractionResult(
                    extracted=False,
                    reason=(
                        "This file is not a Word document, despite its name. "
                        "Upload the .docx saved from Word."
                    ),
                )
        text = _docx_text(data).strip()
    except Exception as exc:
        return ExtractionResult(
            extracted=False,
            reason=f"This Word document could not be opened ({type(exc).__name__}).",
        )

    return _finish(
        text,
        pages=0,
        warnings=[],
        no_text=(
            "This Word document contains no readable text. It may be an empty "
            "template, or the content may be images. Enter the details manually."
        ),
    )


def _finish(
    text: str,
    pages: int,
    warnings: list,
    no_text: str,
) -> ExtractionResult:
    chars = len(text)
    if chars < _MIN_TEXT_LAYER_CHARS:
        return ExtractionResult(
            extracted=False,
            pages=pages,
            chars=chars,
            warnings=warnings,
            reason=no_text,
        )
    return ExtractionResult(
        extracted=True,
        text=text,
        pages=pages,
        chars=chars,
        warnings=warnings,
    )


def extract_text(data: bytes, filename: str = "") -> ExtractionResult:
    """Pull the text out of a PDF or a .docx. Never raises."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadTooLarge(len(data))

    ext = _extension(filename)
    if ext == LEGACY_DOC_EXTENSION:
        raise UnsupportedUpload(
            "Legacy .doc files cannot be read. Open it in Word and use "
            '"Save As" to save it as a .docx, then upload that.'
        )
    if ext and ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise UnsupportedUpload(
            f"Auto-fill accepts PDF and Word (.docx) documents, and this file is {ext}."
        )

    # Dispatch on content, not on the extension. A .doc renamed to .pdf is a zip
    # and would otherwise be handed to pypdf, which fails with a far less
    # helpful message than "this is not a PDF".
    content_kind = None
    if _looks_like_pdf(data):
        content_kind = "pdf"
    elif _looks_like_zip(data):
        content_kind = "zip"

    # A file whose name and content disagree is a common enough mistake — Word
    # saves as .doc, people rename to .pdf to upload it — and answering with
    # "this Word document could not be opened" to someone who believes they
    # uploaded a PDF is not a useful reply. Name both.
    expected_kind = {".pdf": "pdf", ".docx": "zip"}.get(ext)
    if content_kind and expected_kind and content_kind != expected_kind:
        if content_kind == "zip":
            raise UnsupportedUpload(
                "This file is a Word document, but it is named .pdf. Open it in "
                "Word and use \"Save As\" to save it as a .docx, then upload that."
            )
        raise UnsupportedUpload(
            "This file is a PDF, but it is named .docx. Rename it to .pdf and "
            "upload it again."
        )

    if content_kind == "pdf":
        return _pdf_text(data)
    if content_kind == "zip":
        return _docx_result(data)

    if ext == ".docx":
        raise UnsupportedUpload(
            "This file is not a readable Word document, despite its name. "
            "If it was saved by another program, re-save it from Word as .docx."
        )
    # Reported separately from the extension check: a file renamed to .pdf that
    # is really an .xlsx or an image is a different mistake, and saying
    # "got .pdf" back at an officer who uploaded a .pdf reads as a bug.
    raise UnsupportedUpload(
        "This file is not a PDF or a Word document, despite its name. "
        "Re-export it as a PDF or .docx and upload it again."
    )
