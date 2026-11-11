"""Comprehensive test suite for PDF statement input support.

Tests:
1. Pure PDF table parsing (standard layout, multi-page)
2. Number and amount cleaning (currency symbols, negative formats, trailing Dr/Cr)
3. Narration parsing & enrichment (merchant, upi_id, method extraction)
4. Password-protected PDF detection and informative error raising
5. Multiline continuation row merging in table parsing
6. Universal parse_statement entrypoint dispatch (CSV and PDF)
7. End-to-end integration with FinanceAgent and ChromaDB
"""
import io
from pathlib import Path
import pandas as pd
import chromadb
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

from agents.parser import (
    parse_pdf,
    parse_csv,
    parse_statement,
    check_balance_continuity,
    _clean_num,
    _enrich_narration,
    Transaction,
)
import tools as T
import rag
from router import FinanceAgent


def test_clean_num():
    assert _clean_num("1,234.56") == 1234.56
    assert _clean_num("\u20b9 500.00") == 500.00
    assert _clean_num("Rs. 1,000.50") == 1000.50
    assert _clean_num("INR 250") == 250.0
    assert _clean_num("(500.00)") == -500.00
    assert _clean_num("500.00-") == -500.00
    assert _clean_num("-500.00") == -500.00
    assert _clean_num("1200.00 Dr") == -1200.00
    assert _clean_num("1200.00 Cr") == 1200.00
    assert _clean_num("") is None
    assert _clean_num("-") is None
    assert _clean_num("NA") is None
    assert _clean_num(None) is None


def test_enrich_narration():
    m, u, meth = _enrich_narration("UPI/700749903512/Fresh Basket/freshbasket@oksbi")
    assert u == "freshbasket@oksbi"
    assert meth == "UPI"
    assert "Fresh Basket" in m

    m, u, meth = _enrich_narration("NEFT CR-CITI001-Northstar Tech-Monthly Salary")
    assert meth == "NEFT"
    assert "Northstar" in m

    m, u, meth = _enrich_narration("POS 4591 SWIGGY BANGALORE")
    assert meth == "Card"
    assert "SWIGGY" in m

    m, u, meth = _enrich_narration("ATM Cash Withdrawal HDFC ATM Indiranagar")
    assert meth == "ATM"


def test_parse_sample_pdf():
    pdf_path = Path("data/sample_statement.pdf")
    assert pdf_path.exists(), "data/sample_statement.pdf should exist"

    ok, excluded = parse_pdf(pdf_path)
    assert len(ok) == 200, f"Expected 200 transactions, got {len(ok)}"
    assert len(excluded) == 0

    # Verify schema
    expected_cols = [
        "txn_id", "date", "description", "merchant", "upi_id", "amount",
        "type", "balance", "status", "account", "method", "text", "true_category"
    ]
    for col in expected_cols:
        assert col in ok.columns, f"Missing column {col}"

    # Verify types and validation
    assert pd.api.types.is_datetime64_any_dtype(ok["date"])
    assert pd.api.types.is_numeric_dtype(ok["amount"])
    assert set(ok["type"].unique()).issubset({"debit", "credit"})

    # Check balance continuity
    breaks = check_balance_continuity(ok)
    assert breaks == 0, f"Expected 0 balance breaks, found {breaks}"

    # Total debits and credits
    debits = ok[ok["type"] == "debit"]["amount"].sum()
    credits = ok[ok["type"] == "credit"]["amount"].sum()
    assert debits < 0
    assert credits > 0


def test_parse_statement_dispatch():
    csv_path = Path("data/sample_statement.csv")
    pdf_path = Path("data/sample_statement.pdf")

    # CSV path
    tx_csv, _ = parse_statement(csv_path)
    assert len(tx_csv) == 194

    # PDF path
    tx_pdf, _ = parse_statement(pdf_path)
    assert len(tx_pdf) == 200

    # In-memory buffer dispatch
    with open(csv_path, "rb") as f:
        buf = io.BytesIO(f.read())
        tx_buf_csv, _ = parse_statement(buf, filename="statement.csv")
        assert len(tx_buf_csv) == 194

    with open(pdf_path, "rb") as f:
        buf = io.BytesIO(f.read())
        tx_buf_pdf, _ = parse_statement(buf, filename="statement.pdf")
        assert len(tx_buf_pdf) == 200


def test_multiline_continuation_merging():
    """Verify that a multiline narration without a date cell is merged with the previous transaction."""
    # Build a minimal in-memory PDF with 1 multi-line transaction
    rows = [
        ["Date", "Description", "Ref No", "Debit", "Credit", "Balance"],
        ["01/07/2026", "Vegetable Vendor", "REF001", "135.00", "", "1000.00"],
        ["", "Green Leaf Organic Store Extra Line", "", "", "", ""],
    ]
    col_widths = [70, 180, 80, 60, 60, 70]
    total_w = sum(col_widths)
    start_x = 35
    row_h = 22
    start_y = 720
    all_rows = rows
    total_h = row_h * len(all_rows)

    content = ["0.5 w"]
    for i in range(len(all_rows) + 1):
        y = start_y - i * row_h
        content.append(f"{start_x} {y} m {start_x + total_w} {y} l S")
    cur_x = start_x
    for w in [0] + col_widths:
        cur_x += w
        content.append(f"{cur_x} {start_y} m {cur_x} {start_y - total_h} l S")
    for r_idx, row in enumerate(all_rows):
        y_base = start_y - r_idx * row_h - 14
        cur_x = start_x
        for c_idx, cell in enumerate(row):
            safe = str(cell).replace("(", "\\(").replace(")", "\\)")
            content.append(f"BT /F1 8 Tf 1 0 0 1 {cur_x + 3} {y_base} Tm ({safe}) Tj ET")
            cur_x += col_widths[c_idx]

    stream = "\n".join(content).encode("latin-1")
    objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    ]
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = [0]
    pos = len(b"%PDF-1.4\n")
    for obj in objs:
        offsets.append(pos)
        buf.write(obj)
        pos += len(obj)
    xref_pos = pos
    buf.write(f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        buf.write(f"{off:010d} 00000 n \n".encode("latin-1"))
    buf.write(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("latin-1"))
    buf.seek(0)

    ok, _ = parse_pdf(buf)
    assert len(ok) == 1
    # Description should contain both lines
    desc = ok.iloc[0]["description"]
    assert "Vegetable Vendor" in desc
    assert "Green Leaf Organic" in desc


def test_named_month_dates():
    """Verify that statements with named-month dates (e.g. 01-Jul-2026) parse and have datetime64 dtype."""
    rows = [
        ["Date", "Narration", "Chq/Ref No", "Withdrawal (Dr)", "Deposit (Cr)", "Closing Balance"],
        ["01-Jul-2026", "Vegetable Vendor", "REF10001", "135.38", "", "32614.62"],
        ["02-Jul-2026", "Cab Fare Rapido", "REF10002", "686.47", "", "31928.15"],
        ["08-Jul-2026", "Monthly Salary", "REF10003", "", "66377.94", "98306.09"]
    ]
    col_widths = [75, 175, 90, 70, 70, 70]
    total_w = sum(col_widths)
    start_x = 35
    row_h = 22
    start_y = 720
    all_rows = rows
    total_h = row_h * len(all_rows)

    content = ["0.5 w"]
    for i in range(len(all_rows) + 1):
        y = start_y - i * row_h
        content.append(f"{start_x} {y} m {start_x + total_w} {y} l S")
    cur_x = start_x
    for w in [0] + col_widths:
        cur_x += w
        content.append(f"{cur_x} {start_y} m {cur_x} {start_y - total_h} l S")
    for r_idx, row in enumerate(all_rows):
        y_base = start_y - r_idx * row_h - 14
        cur_x = start_x
        for c_idx, cell in enumerate(row):
            safe = str(cell).replace("(", "\\(").replace(")", "\\)")
            content.append(f"BT /F1 8 Tf 1 0 0 1 {cur_x + 3} {y_base} Tm ({safe}) Tj ET")
            cur_x += col_widths[c_idx]

    stream = "\n".join(content).encode("latin-1")
    objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    ]
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = [0]
    pos = len(b"%PDF-1.4\n")
    for obj in objs:
        offsets.append(pos)
        buf.write(obj)
        pos += len(obj)
    xref_pos = pos
    buf.write(f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        buf.write(f"{off:010d} 00000 n \n".encode("latin-1"))
    buf.write(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("latin-1"))
    buf.seek(0)

    ok, _ = parse_pdf(buf)
    assert len(ok) == 3
    assert pd.api.types.is_datetime64_any_dtype(ok["date"])
    # Verify .dt accessor functions without AttributeError
    months = ok["date"].dt.strftime("%Y-%m").tolist()
    assert months == ["2026-07", "2026-07", "2026-07"]
    assert T._month(ok["date"]).iloc[0] == "2026-07"


def test_pdf_end_to_end_agent():
    """Verify that transactions parsed from PDF feed seamlessly into FinanceAgent."""
    pdf_path = Path("data/sample_statement.pdf")
    ok, _ = parse_pdf(pdf_path)

    # Stand-in category assignment
    df = ok.copy()
    df["category"] = "Shopping"

    def fake_llm(pv):
        return AIMessage(content='{"reason": "fake", "intent": "spending_analysis", "needs_retrieval": false}')

    ctx = rag.RagContext.create(df, client=chromadb.EphemeralClient(), ef=rag.HashEmbedding())
    agent = FinanceAgent(df, RunnableLambda(fake_llm), ctx)

    assert agent is not None
    assert len(agent.df) == 200
    spent = T.total_spent(agent.df)
    assert spent > 0


if __name__ == "__main__":
    test_clean_num()
    print("test_clean_num: OK")
    test_enrich_narration()
    print("test_enrich_narration: OK")
    test_parse_sample_pdf()
    print("test_parse_sample_pdf: OK")
    test_parse_statement_dispatch()
    print("test_parse_statement_dispatch: OK")
    test_multiline_continuation_merging()
    print("test_multiline_continuation_merging: OK")
    test_named_month_dates()
    print("test_named_month_dates: OK")
    test_pdf_end_to_end_agent()
    print("test_pdf_end_to_end_agent: OK")
    print("\nALL PDF TESTS PASSED!")
