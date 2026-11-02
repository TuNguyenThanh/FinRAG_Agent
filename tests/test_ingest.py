import pytest

from app.rag.ingest import UnsupportedFileError, chunk_document

MD = """# Biểu phí

## Chuyển khoản
Phí chuyển khoản nội bộ miễn phí trên mọi kênh giao dịch điện tử.

## Thẻ
### Thẻ Gold
Phí thường niên thẻ Gold là 500.000 đồng, miễn phí năm đầu.
"""


def test_markdown_chunks_keep_section_path():
    chunks = chunk_document("bieu-phi.md", MD.encode(), 800, 100)
    sections = {c.section for c in chunks}
    assert "Chuyển khoản" in sections
    assert "Thẻ > Thẻ Gold" in sections
    assert all(c.doc_name == "bieu-phi.md" for c in chunks)


def test_long_text_is_split_with_size_limit():
    text = ("Điều khoản sử dụng dịch vụ ngân hàng điện tử. " * 200).encode()
    chunks = chunk_document("terms.txt", text, 500, 50)
    assert len(chunks) > 5
    assert all(len(c.content) <= 500 for c in chunks)


def test_rejects_unknown_type():
    with pytest.raises(UnsupportedFileError):
        chunk_document("x.exe", b"MZ", 500, 50)
