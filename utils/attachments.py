"""
Turn chat-input attachments into something the tutor can actually read.

Text files (txt/csv/tsv/md/json) and PDFs are decoded and inlined into the
student's query. Screenshots are decoded, downscaled, and handed back
separately as base64 data URLs for the vision builds of the tutoring chains
(see `chains_lcel._with_vision` and `ta_tools.StreamSpec.images`).

Images deliberately do NOT ride along in the query string. The router only has
to know that a screenshot exists in order to pick a tool; paying image tokens
on every hop of the agent loop buys nothing, and the tool-call arguments are
the wrong place to carry a megabyte of base64.
"""

import base64
from io import BytesIO
from typing import Dict, List, Tuple

# Per-file and total budgets. A pasted JMP export can be enormous, and the
# tutoring prompts are already carrying course context plus chat history.
MAX_CHARS_PER_FILE = 4000
MAX_CHARS_TOTAL = 8000
MAX_PDF_PAGES = 12

# Vision models downscale anything larger than this internally, so sending more
# pixels costs tokens and latency without adding a detail the model can use.
MAX_IMAGE_EDGE = 1568
# Three screenshots is already an unusual turn; past that the student is
# uploading an assignment, not asking about their work.
MAX_IMAGES = 3

IMAGE_LIMIT_NOTICE = (
    f"I only looked at the first {MAX_IMAGES} screenshots."
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


def _downscale(raw: bytes, mime: str) -> Tuple[bytes, str]:
    """Shrink an oversized screenshot; pass a small one through untouched.

    Re-encodes as PNG rather than JPEG. These are pictures of regression
    tables and dialog boxes, and JPEG ringing around 9pt text is exactly the
    artefact that turns `p = 0.004` into a misread coefficient.

    Pillow arrives with Streamlit, but every failure here is recoverable by
    sending the original bytes, so nothing in this function raises.
    """
    try:
        from PIL import Image
    except ImportError:
        return raw, mime

    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception:
        return raw, mime

    already_small = max(image.size) <= MAX_IMAGE_EDGE
    if already_small and mime in ("image/png", "image/jpeg", "image/webp"):
        return raw, mime

    try:
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.LANCZOS)
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"
    except Exception:
        return raw, mime


def _to_image_part(upload) -> Dict[str, str]:
    """One screenshot as `{name, data_url}`, or `{}` if it could not be read."""
    try:
        raw = upload.getvalue()
    except Exception:
        return {}
    if not raw:
        return {}

    mime = (getattr(upload, "type", "") or "").lower() or "image/png"
    data, mime = _downscale(raw, mime)
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "name": getattr(upload, "name", "screenshot"),
        "data_url": f"data:{mime};base64,{encoded}",
    }


def describe_images(images: List[Dict[str, str]]) -> str:
    """The marker the ROUTER sees in place of the images themselves.

    It has to carry two facts and no more: a screenshot is present, and the
    tutor can actually see it. Without the second half the router used to pick
    a route that apologises for not reading images.
    """
    if not images:
        return ""
    single = len(images) == 1
    subject = "A screenshot" if single else f"{len(images)} screenshots"
    verb = "is" if single else "are"
    pronoun = "it" if single else "them"
    # No filenames here -- app.py already names every attachment on the line
    # above this one, and saying it twice just spends context.
    return (
        f"\n\n[{subject} of the student's own work {verb} attached and visible "
        f"to you: whichever tool you choose will be shown {pronoun}. Do not "
        "tell the student you cannot read images.]"
    )


ATTACHMENT_HEADER = (
    "\n\nThe student attached the following file content. Treat it as their "
    "own work or data, not as course material:\n\n"
)


def attachment_query_block(attachment_text: str) -> str:
    """The attached file content as it is appended to the ROUTER's query.

    Kept separate from the raw blocks so the same text can be handed to
    check_attempt without the framing sentence, which is addressed to the
    router and would otherwise be graded as part of the attempt.
    """
    return ATTACHMENT_HEADER + attachment_text if attachment_text else ""


def extract_attachments(uploads) -> Tuple[str, List[str], List[Dict[str, str]]]:
    """Read what can be read.

    Returns `(attachment_text, unreadable_names, images)`.

    `attachment_text` is the decoded file content, one "--- Attached file ---"
    block per file and nothing else. It goes to the router inside
    attachment_query_block() and to check_attempt as-is. `images` is handed to
    the tools out-of-band. `unreadable_names` drives the UI notice, so the
    student learns immediately that a file was not consulted rather than
    inferring it from a vague answer.
    """
    if not uploads:
        return "", [], []

    blocks: List[str] = []
    unreadable: List[str] = []
    images: List[Dict[str, str]] = []
    budget = MAX_CHARS_TOTAL

    for upload in uploads:
        name = getattr(upload, "name", "attachment")
        if _is_image(upload):
            if len(images) >= MAX_IMAGES:
                unreadable.append(name)
                continue
            part = _to_image_part(upload)
            if part:
                images.append(part)
            else:
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

    return "\n\n".join(blocks), unreadable, images
