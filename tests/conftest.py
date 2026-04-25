"""Shared pytest fixtures and helpers."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory (provided by pytest)."""
    return tmp_path


@pytest.fixture()
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal valid PDF and return its path.

    Uses reportlab if available; falls back to writing a raw minimal PDF
    byte sequence that pdfplumber can parse.
    """
    pdf_path = tmp_path / "sample.pdf"
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(72, 720, "Hello World – Page 1")
        c.showPage()
        c.drawString(72, 720, "Second page content about enterprise RAG.")
        c.showPage()
        c.save()
    except ImportError:
        # Minimal valid 1-page PDF (no images, just text via raw content stream)
        pdf_bytes = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            b"4 0 obj<</Length 44>>\nstream\nBT /F1 12 Tf 72 720 Td (Hello World) Tj ET\nendstream\nendobj\n"
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000274 00000 n \n"
            b"0000000370 00000 n \n"
            b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n441\n%%EOF\n"
        )
        pdf_path.write_bytes(pdf_bytes)
    return pdf_path
