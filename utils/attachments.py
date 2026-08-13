"""
Turn chat-input attachments into text the tutor can actually read.

Until this module existed the app accepted png/jpg/pdf/txt/csv uploads,
displayed them in the student's message bubble, and then sent the model only
the file NAMES:

    [Student attached: regression_output.png]

So "check my work" against an uploaded file was answered from the filename.
The UI promised something the backend never delivered, which is worse than not
offering uploads at all -- the student believes their work was reviewed.

What is fixed here: txt, csv, and pdf are read and inlined. What is NOT fixed:
images. Every tutoring chain is a text template (see chains_lcel.py), so
feeding a screenshot to the model means restructuring the payloads for
multimodal messages. Until that happens, images are reported honestly rather
than silently dropped -- see IMAGE_NOTICE.
"""

from typing import List, Tuple

# Per-file and total budgets. A pasted JMP export can be enormous, and the
# tutoring prompts are already carrying course context plus chat history.
MAX_CHARS_PER_FILE = 4000
MAX_CHARS_TOTAL = 8000
MAX_PDF_PAGES = 12

IMAGE_NOTICE = (
    "I can't read images yet, so I have not looked at that screenshot. "
    "Paste the numbers or the output text and I will work from that."
)


def _is_image(upload) -> bool:
    return (getattr(upload, "type", "") or "").lower().startswith("image/")


def _read_text(upload) -> str:
    raw = upload.getvalue()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return ""


def _read_pdf(upload) -> str:
    try:
        from io import BytesIO

        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(BytesIO(upload.getvalue()))
        pages = [(p.extract_text() or "") for p in reader.pages[:MAX_PDF_PAGES]]
        return "\n".join(pages)
    except Exception:
        # A corrupt or image-only PDF should cost the student a note, not the turn.
        return ""


def _extract_one(upload) -> str:
    name = (getattr(upload, "name", "") or "").lower()
    if name.endswith(".pdf"):
        text = _read_pdf(upload)
    elif name.endswith((".txt", ".csv", ".tsv", ".md", ".json")):
        text = _read_text(upload)
    else:
        text = ""
    text = (text or "").strip()
    if len(text) > MAX_CHARS_PER_FILE:
        text = text[:MAX_CHARS_PER_FILE] + "\n[...truncated]"
    return text


def extract_attachments(uploads) -> Tuple[str, List[str], bool]:
    """Read what can be read.

    Returns (context_block, unreadable_names, has_image).

    `context_block` is appended to the student's query. `unreadable_names` and
    `has_image` drive the UI notice, so the student learns immediately that a
    screenshot was not consulted rather than inferring it from a vague answer.
    """
    if not uploads:
        return "", [], False

    blocks: List[str] = []
    unreadable: List[str] = []
    has_image = False
    budget = MAX_CHARS_TOTAL

    for upload in uploads:
        name = getattr(upload, "name", "attachment")
        if _is_image(upload):
            has_image = True
            unreadable.append(name)
            continue
        text = _extract_one(upload)
        if not text:
            unreadable.append(name)
            continue
        if len(text) > budget:
            text = text[:budget] + "\n[...truncated]"
        budget -= len(text)
        blocks.append(f"--- Attached file: {name} ---\n{text}")
        if budget <= 0:
            break

    if not blocks:
        return "", unreadable, has_image

    header = (
        "\n\nThe student attached the following file content. Treat it as their "
        "own work or data, not as course material:\n\n"
    )
    return header + "\n\n".join(blocks), unreadable, has_image
