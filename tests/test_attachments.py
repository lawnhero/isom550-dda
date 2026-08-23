"""Attachment extraction, and the split between what the router sees and what
check_attempt receives."""

from io import BytesIO

from utils.attachments import (
    ATTACHMENT_HEADER,
    attachment_query_block,
    describe_images,
    extract_attachments,
)


class _Upload:
    def __init__(self, name, data: bytes, mime="text/plain"):
        self.name = name
        self.type = mime
        self._data = data

    def getvalue(self):
        return self._data


def test_text_attachment_is_a_bare_block_without_the_router_header():
    text, unreadable, images = extract_attachments(
        [_Upload("work.txt", b"R-squared = 0.62\nslope = 2.1")]
    )
    assert text.startswith("--- Attached file: work.txt ---")
    assert "R-squared = 0.62" in text
    assert ATTACHMENT_HEADER.strip() not in text
    assert unreadable == [] and images == []


def test_query_block_adds_the_header_only_when_there_is_content():
    assert attachment_query_block("") == ""
    block = attachment_query_block("--- Attached file: a.txt ---\nhello")
    assert block.startswith(ATTACHMENT_HEADER)
    assert block.endswith("hello")


def test_unreadable_files_are_reported_not_silently_dropped():
    text, unreadable, images = extract_attachments(
        [_Upload("model.xlsx", b"\x00\x01", mime="application/octet-stream")]
    )
    assert text == ""
    assert unreadable == ["model.xlsx"]


def test_images_go_out_of_band_and_are_described_to_the_router():
    # A 1x1 PNG.
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfe\xa7V\xbd\xfa\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    text, unreadable, images = extract_attachments([_Upload("shot.png", png, mime="image/png")])
    assert text == "" and unreadable == []
    assert len(images) == 1 and images[0]["data_url"].startswith("data:image/png;base64,")
    marker = describe_images(images)
    assert "A screenshot" in marker and "Do not tell the student you cannot read images" in marker
    assert describe_images([]) == ""
