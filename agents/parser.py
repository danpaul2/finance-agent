"""Parser Agent: raw bank statement (CSV or PDF) -> fixed schema.

Output schema (one row per transaction):
    txn_id, date, description, merchant, upi_id, amount (signed: debit<0, credit>0),
    type ('debit'|'credit'), balance, status, account, method, text (for categorizer/embeddings)

Failed/reversed rows are separated out so they never pollute totals.
The original `category` column (if any) is kept as `true_category` for evaluation only.
"""
from __future__ import annotations
import io
import re
from datetime import date as Date
from pathlib import Path
import pandas as pd
import pdfplumber
from pdfminer.pdfdocument import PDFEncryptionError, PDFPasswordIncorrect
from pydantic import BaseModel, field_validator

# Maps common bank column names -> our schema. Unknown headers can later be mapped by an LLM fallback.
COLUMN_ALIASES = {
    "txn_id": ["transaction_id", "txn_id", "id", "reference_id", "ref no", "chq./ref.no.", "chq/ref no",
               "chq no", "cheque no", "ref no.", "reference", "utr", "txn ref", "tran id", "chq / ref no", "ref"],
    "date": ["transaction_date", "date", "txn date", "value date", "tran date", "trans date",
             "posting date", "txn dt", "value dt", "trans. date", "txn. date", "txndate", "valdate", "post date"],
    "time": ["transaction_time", "time"],
    "description": ["description", "narration", "particulars", "remarks", "transaction remarks",
                    "transaction details", "details", "particular", "transaction narration"],
    "merchant": ["merchant_or_counterparty", "merchant", "counterparty", "payee", "beneficiary"],
    "debit": ["debit_inr", "debit", "withdrawal", "withdrawal amt", "withdrawal amt.",
              "withdrawal amount", "withdrawal (dr)", "dr", "dr.", "dr amount", "withdrawals",
              "debit (inr)", "debit (rs.)", "withdrawals (dr)", "debits"],
    "credit": ["credit_inr", "credit", "deposit", "deposit amt", "deposit amt.",
               "deposit amount", "deposit (cr)", "cr", "cr.", "cr amount", "deposits",
               "credit (inr)", "credit (rs.)", "deposits (cr)", "credits"],
    "amount": ["amount", "txn amount", "transaction amount", "net amount", "amount (inr)", "amount (rs.)"],
    "balance": ["balance_inr", "balance", "closing balance", "bal", "running balance", "closing bal",
                "balance (inr)", "balance (rs.)", "closing balance (inr)"],
    "status": ["status"],
    "account": ["account_label", "account", "account no", "account number"],
    "method": ["payment_method", "mode", "method", "type of payment", "channel"],
    "upi_id": ["upi_id", "vpa", "upi ref", "upi id"],
    "true_category": ["category"],
}


class Transaction(BaseModel):
    """Pydantic validation of each parsed row."""
    txn_id: str
    date: Date
    description: str
    amount: float
    type: str

    @field_validator("type")
    @classmethod
    def _type_ok(cls, v):
        if v not in ("debit", "credit"):
            raise ValueError("type must be debit or credit")
        return v


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for std, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                mapping[lower[a]] = std
                break
    return df.rename(columns=mapping)


def _parse_single_date(val: object) -> pd.Timestamp | None:
    """Validate and parse a single date string flexibly (supports ISO, named months, dotted dates)."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if len(s) < 6 or len(s) > 35 or not any(c.isdigit() for c in s):
        return None
    # Exclude monetary numbers (e.g. 135.38, 1,234.50, 1000)
    if re.match(r"^[\d,]+(?:\.\d{1,2})?$", s):
        return None
    # Convert dotted dd.mm.yyyy to dd-mm-yyyy
    s_clean = re.sub(r"^(\d{1,2})\.(\d{1,2})\.(\d{2,4})", r"\1-\2-\3", s)
    try:
        dt = pd.to_datetime(s_clean, format="mixed", dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return dt
        # Try first token if timestamp appended e.g. '01-Jul-2026 14:30:00'
        first_token = s_clean.split()[0]
        dt = pd.to_datetime(first_token, format="mixed", dayfirst=True, errors="coerce")
        return dt if pd.notna(dt) else None
    except Exception:
        return None


def _is_valid_date(val: object) -> bool:
    return _parse_single_date(val) is not None


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse dates handling ISO (YYYY-MM-DD), Indian DD/MM/YYYY, named months, and timestamps."""
    if s.empty:
        return pd.Series([], dtype="datetime64[ns]")
    cleaned = s.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"^(\d{1,2})\.(\d{1,2})\.(\d{2,4})", r"\1-\2-\3", regex=True)
    res = pd.to_datetime(cleaned, format="mixed", dayfirst=True, errors="coerce")
    return pd.to_datetime(res)


def _clean_num(val: object) -> float | None:
    """Clean monetary strings (remove commas, currency symbols, handle trailing Dr/Cr or negatives)."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().replace(",", "").replace("\u20b9", "").replace("Rs.", "").replace("INR", "").strip()
    if not s or s in ("-", "--", "NA", "N/A", "Nil", "nil", "none"):
        return None
    sign = 1.0
    if s.startswith("(") and s.endswith(")"):
        sign = -1.0
        s = s[1:-1].strip()
    elif s.endswith("-"):
        sign = -1.0
        s = s[:-1].strip()
    elif s.startswith("-"):
        sign = -1.0
        s = s[1:].strip()
    elif s.upper().endswith("DR"):
        sign = -1.0
        s = s[:-2].strip()
    elif s.upper().endswith("CR"):
        sign = 1.0
        s = s[:-2].strip()
    try:
        return sign * float(s)
    except ValueError:
        return None


def _enrich_narration(desc: str) -> tuple[str, str | None, str | None]:
    """Extract (merchant, upi_id, method) from a free-text bank narration when columns are missing."""
    if not desc:
        return "", None, None
    text = desc.strip()

    # Extract UPI ID
    upi_match = re.search(r"([a-zA-Z0-9.\-_]+@[a-zA-Z0-9]+)", text)
    upi_id = upi_match.group(1) if upi_match else None

    # Detect payment method
    method = None
    u = text.upper()
    if upi_id or "UPI" in u:
        method = "UPI"
    elif "NEFT" in u:
        method = "NEFT"
    elif "IMPS" in u:
        method = "IMPS"
    elif "RTGS" in u:
        method = "RTGS"
    elif "ATM" in u or "CASH WDL" in u:
        method = "ATM"
    elif "POS" in u or "CARD" in u:
        method = "Card"
    elif "BBPS" in u or "BILL" in u:
        method = "BBPS"
    elif "NETBANKING" in u or "TRANSFER" in u:
        method = "NetBanking"

    # Extract merchant name
    merchant = ""
    if "UPI/" in u or "UPI-" in u:
        parts = re.split(r"[/\\-]", text)
        for p in parts[1:]:
            p_clean = p.strip()
            if (p_clean and not re.match(r"^\d+$", p_clean) and "@" not in p_clean and
                    p_clean.upper() not in ("DR", "CR", "UPI", "OKSBI", "OKAXIS", "OKICICI", "PAYTM")):
                merchant = p_clean
                break
    elif u.startswith("NEFT") or u.startswith("IMPS"):
        parts = re.split(r"[-/]", text)
        for p in parts[1:]:
            p_clean = p.strip()
            if p_clean and not re.match(r"^[A-Z0-9]+$", p_clean):
                merchant = p_clean
                break
    elif u.startswith("POS "):
        tokens = text.split()
        merchant = " ".join([t for t in tokens[1:] if not re.match(r"^\d+$", t)])

    if not merchant:
        cleaned = re.sub(r"^(UPI[/ -]|NEFT[ -]|IMPS[ -]|POS \d+ |ATM )", "", text, flags=re.I).strip()
        merchant = cleaned if cleaned else text

    return merchant, upi_id, method


def _match_header_cell(cell: str | None) -> str | None:
    if not cell:
        return None
    raw = str(cell).lower().strip()
    cl = re.sub(r"[\s_/\\()\[\].-]+", " ", raw).strip()
    for std, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            a_norm = re.sub(r"[\s_/\\()\[\].-]+", " ", a).strip()
            if cl == a_norm or re.search(rf"\b{re.escape(a_norm)}\b", cl):
                return std
    return None


def _is_header_row(row: list) -> dict[int, str] | None:
    matched = {}
    for idx, cell in enumerate(row):
        std = _match_header_cell(cell)
        if std and std not in matched.values():
            matched[idx] = std
    has_date = "date" in matched.values()
    has_data = any(c in matched.values() for c in ("description", "debit", "credit", "amount", "balance"))
    if has_date and (has_data or len(matched) >= 3):
        return matched
    return None


def _is_summary_or_footer(row: list) -> bool:
    text = " ".join(str(c or "") for c in row).strip().lower()
    if not text:
        return True
    patterns = [
        r"^page \d+",
        r"^(opening balance|brought forward|b/f|carried forward|c/f)",
        r"^(total|grand total|subtotal)\b",
        r"^(statement summary|account summary)",
        r"^(closing balance|closing bal)\b",
    ]
    return any(re.search(p, text) for p in patterns)


def parse_csv(path_or_buffer) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, excluded). `excluded` holds failed/reversed rows."""
    raw = pd.read_csv(path_or_buffer)
    df = _rename(raw)

    if "debit" in df.columns or "credit" in df.columns:
        for col in ("debit", "credit"):
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
        df["amount"] = df["credit"] - df["debit"]
        df["type"] = df["amount"].apply(lambda x: "credit" if x > 0 else "debit")
    elif "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        if "type" in df.columns:
            df["type"] = df["type"].astype(str).str.lower().apply(lambda t: "debit" if "dr" in t or "debit" in t else "credit")
            df["amount"] = df.apply(lambda r: -abs(r["amount"]) if r["type"] == "debit" else abs(r["amount"]), axis=1)
        else:
            df["type"] = df["amount"].apply(lambda x: "credit" if x > 0 else "debit")
    else:
        df["amount"] = 0.0
        df["type"] = "debit"

    df["date"] = _parse_dates(df["date"])

    for col in ("merchant", "upi_id", "method", "account", "status", "true_category", "txn_id"):
        if col not in df:
            df[col] = None
    df["status"] = df["status"].fillna("SUCCESS").str.upper()
    df["description"] = df["description"].fillna("").astype(str)
    df["merchant"] = df["merchant"].fillna("").astype(str)

    # Fill missing merchant, upi_id, or method if possible
    for idx, row in df.iterrows():
        if not row["merchant"] or not row["method"]:
            m, u, meth = _enrich_narration(row["description"])
            if not row["merchant"]:
                df.at[idx, "merchant"] = m
            if not row["upi_id"] and u:
                df.at[idx, "upi_id"] = u
            if not row["method"] and meth:
                df.at[idx, "method"] = meth
        if not row["txn_id"]:
            df.at[idx, "txn_id"] = f"TXN{idx+1:06d}"

    # Single text field used by the categorizer prompt and ChromaDB embeddings
    df["text"] = (df["description"] + " | " + df["merchant"] + " | "
                  + df["upi_id"].fillna("").astype(str) + " | " + df["method"].fillna("").astype(str)).str.strip(" |")

    bad = df["status"].ne("SUCCESS") | df["date"].isna()
    excluded, ok = df[bad].copy(), df[~bad].copy()

    # Ensure date column has datetime64 dtype
    ok["date"] = pd.to_datetime(ok["date"])
    excluded["date"] = pd.to_datetime(excluded["date"])

    # Validate every kept row against the schema
    for r in ok.itertuples():
        d = r.date.date() if hasattr(r.date, "date") else pd.to_datetime(r.date).date()
        Transaction(txn_id=str(r.txn_id), date=d, description=r.description,
                    amount=r.amount, type=r.type)

    keep = ["txn_id", "date", "description", "merchant", "upi_id", "amount", "type",
            "balance", "status", "account", "method", "text", "true_category"]
    return ok[keep].reset_index(drop=True), excluded[keep].reset_index(drop=True)


def _extract_rows_from_tables(pdf, table_settings=None) -> list[dict[str, str]]:
    """Extract rows across pages using pdfplumber table extraction."""
    all_rows: list[dict[str, str]] = []
    current_col_mapping: dict[int, str] | None = None

    for page in pdf.pages:
        tables = page.extract_tables(table_settings) if table_settings else page.extract_tables()
        if not tables:
            continue

        for table in tables:
            for row in table:
                if not row or all(not str(c or "").strip() for c in row):
                    continue

                header_map = _is_header_row(row)
                if header_map:
                    current_col_mapping = header_map
                    continue

                if current_col_mapping is None:
                    continue

                if _is_summary_or_footer(row):
                    continue

                row_data: dict[str, str] = {}
                for col_idx, std_name in current_col_mapping.items():
                    if col_idx < len(row):
                        row_data[std_name] = str(row[col_idx] or "").strip()
                    else:
                        row_data[std_name] = ""

                date_val = row_data.get("date", "")
                if not _is_valid_date(date_val):
                    # Multiline continuation
                    desc = row_data.get("description", "")
                    if not desc:
                        desc = " ".join(str(c or "").strip() for c in row if str(c or "").strip())
                    if desc and all_rows:
                        prev = all_rows[-1].get("description", "")
                        all_rows[-1]["description"] = (prev + " " + desc).strip()
                    continue

                all_rows.append(row_data)

    return all_rows


def _extract_rows_from_text(pdf) -> list[dict[str, str]]:
    """Fallback: extract transactions line-by-line from raw text for borderless/unstructured PDFs."""
    all_rows: list[dict[str, str]] = []
    header_found = False

    for page in pdf.pages:
        raw_text = page.extract_text()
        if not raw_text:
            continue

        for line in raw_text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            if not header_found:
                lower = line_str.lower()
                if "date" in lower and any(k in lower for k in ("narration", "description", "particular", "withdrawal", "debit", "deposit", "credit", "amount", "balance")):
                    header_found = True
                continue

            if _is_summary_or_footer([line_str]):
                continue

            # Look for lines starting with a date
            m = re.match(r"^(\d{1,4}[-/.]\w{1,3}[-/.]\d{1,4}|\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\w{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+\w{3,9}\s+\d{4})\s+(.+)$", line_str)
            if not m:
                # Continuation line
                if all_rows and not re.match(r"^(page\s+\d+|total\b)", line_str, re.I):
                    all_rows[-1]["description"] = (all_rows[-1]["description"] + " " + line_str).strip()
                continue

            dt_str, rest = m.group(1), m.group(2).strip()
            if not _is_valid_date(dt_str):
                continue

            # Extract numbers from right: [withdrawal, deposit, balance] or [amount, balance]
            tokens = rest.split()
            nums = []
            desc_tokens = []
            for i in range(len(tokens) - 1, -1, -1):
                tok_clean = tokens[i].replace(",", "")
                if re.match(r"^-?\d+(?:\.\d+)?(?:cr|dr)?$", tok_clean, re.I):
                    nums.insert(0, tokens[i])
                    if len(nums) == 3:
                        desc_tokens = tokens[:i]
                        break
                else:
                    desc_tokens = tokens[:i + 1]
                    break

            desc = " ".join(desc_tokens) if desc_tokens else rest
            row_data = {"date": dt_str, "description": desc}
            if len(nums) == 3:
                row_data["debit"] = nums[0]
                row_data["credit"] = nums[1]
                row_data["balance"] = nums[2]
            elif len(nums) == 2:
                # amount and balance
                row_data["amount"] = nums[0]
                row_data["balance"] = nums[1]
            elif len(nums) == 1:
                row_data["amount"] = nums[0]

            all_rows.append(row_data)

    return all_rows


def parse_pdf(path_or_buffer, password: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a bank statement from a PDF file or buffer.

    Supports:
    - Grid-bordered tables (standard bank layout)
    - Borderless columnar tables (text-aligned layout)
    - Line-by-line unstructured text fallback
    - Multiline narrations spanning table rows
    - Password-protected bank PDFs (with descriptive error on missing/wrong password)
    """
    try:
        pdf = pdfplumber.open(path_or_buffer, password=password)
    except (PDFPasswordIncorrect, PDFEncryptionError) as e:
        raise ValueError("This PDF statement is password-protected. Please provide the correct password.") from e
    except Exception as e:
        err = str(e).lower()
        if "password" in err or "encrypt" in err or "protected" in err:
            raise ValueError("This PDF statement is password-protected. Please provide the correct password.") from e
        raise

    keep = ["txn_id", "date", "description", "merchant", "upi_id", "amount", "type",
            "balance", "status", "account", "method", "text", "true_category"]

    with pdf:
        # Strategy 1: Grid table extraction
        all_rows = _extract_rows_from_tables(pdf)

        # Strategy 2: Whitespace / text table extraction
        if not all_rows:
            all_rows = _extract_rows_from_tables(pdf, table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})

        # Strategy 3: Line-by-line text parsing
        if not all_rows:
            all_rows = _extract_rows_from_text(pdf)

    if not all_rows:
        empty = pd.DataFrame(columns=keep)
        empty["date"] = pd.to_datetime(empty["date"])
        return empty, empty

    df = pd.DataFrame(all_rows)

    # Process amounts and debit/credit directions
    if "debit" in df.columns or "credit" in df.columns:
        df["debit"] = pd.to_numeric(df.get("debit", 0).apply(_clean_num), errors="coerce").fillna(0.0).abs()
        df["credit"] = pd.to_numeric(df.get("credit", 0).apply(_clean_num), errors="coerce").fillna(0.0).abs()
        df["amount"] = df["credit"] - df["debit"]
        df["type"] = df["amount"].apply(lambda x: "credit" if x > 0 else "debit")
    elif "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"].apply(_clean_num), errors="coerce").fillna(0.0)
        if "type" in df.columns:
            df["type"] = df["type"].astype(str).str.lower().apply(lambda t: "debit" if "dr" in t or "debit" in t else "credit")
            df["amount"] = df.apply(lambda r: -abs(r["amount"]) if r["type"] == "debit" else abs(r["amount"]), axis=1)
        else:
            df["type"] = df["amount"].apply(lambda x: "credit" if x > 0 else "debit")
    else:
        df["amount"] = 0.0
        df["type"] = "debit"

    df["date"] = _parse_dates(df["date"])

    if "balance" in df.columns:
        df["balance"] = pd.to_numeric(df["balance"].apply(_clean_num), errors="coerce")
    else:
        df["balance"] = None

    for col in ("merchant", "upi_id", "method", "account", "status", "true_category", "txn_id"):
        if col not in df:
            df[col] = None

    df["status"] = df["status"].fillna("SUCCESS").str.upper()
    df["description"] = df["description"].fillna("").astype(str).str.replace("\n", " ").str.strip()

    # Enrich merchant, upi_id, and method from narration if absent
    for idx, row in df.iterrows():
        d = row["description"]
        m, u, meth = _enrich_narration(d)
        if not row["merchant"]:
            df.at[idx, "merchant"] = m
        if not row["upi_id"]:
            df.at[idx, "upi_id"] = u
        if not row["method"]:
            df.at[idx, "method"] = meth
        if not row["txn_id"]:
            df.at[idx, "txn_id"] = f"TXN{idx+1:06d}"

    df["merchant"] = df["merchant"].fillna("").astype(str)

    # Single text field used by categorizer prompt and ChromaDB embeddings
    df["text"] = (df["description"] + " | " + df["merchant"] + " | "
                  + df["upi_id"].fillna("").astype(str) + " | " + df["method"].fillna("").astype(str)).str.strip(" |")

    bad = df["status"].ne("SUCCESS") | df["date"].isna()
    excluded, ok = df[bad].copy(), df[~bad].copy()

    # Ensure date column has datetime64 dtype
    ok["date"] = pd.to_datetime(ok["date"])
    excluded["date"] = pd.to_datetime(excluded["date"])

    # Schema validation against Pydantic model
    for r in ok.itertuples():
        d = r.date.date() if hasattr(r.date, "date") else pd.to_datetime(r.date).date()
        Transaction(txn_id=str(r.txn_id), date=d, description=r.description,
                    amount=r.amount, type=r.type)

    return ok[keep].reset_index(drop=True), excluded[keep].reset_index(drop=True)


def parse_statement(path_or_buffer, filename: str | None = None, password: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Universal bank statement entrypoint: auto-detects CSV or PDF and parses accordingly."""
    fname = filename or getattr(path_or_buffer, "name", "")
    if isinstance(fname, Path):
        fname = str(fname)
    fname_lower = str(fname).lower() if fname else ""

    is_pdf = False
    if fname_lower.endswith(".pdf"):
        is_pdf = True
    elif fname_lower.endswith(".csv"):
        is_pdf = False
    elif isinstance(path_or_buffer, (str, Path)):
        is_pdf = str(path_or_buffer).lower().endswith(".pdf")
    elif hasattr(path_or_buffer, "seek") and hasattr(path_or_buffer, "read"):
        start = path_or_buffer.read(5)
        path_or_buffer.seek(0)
        if start.startswith(b"%PDF"):
            is_pdf = True

    if is_pdf:
        return parse_pdf(path_or_buffer, password=password)
    return parse_csv(path_or_buffer)


def check_balance_continuity(df: pd.DataFrame) -> int:
    """Number of rows where balance != previous balance + amount (first row skipped)."""
    d = (df["balance"].diff() - df["amount"]).iloc[1:]
    return int((d.abs() > 0.01).sum())


if __name__ == "__main__":
    tx, ex = parse_csv("data/sample_statement.csv")
    print(f"CSV: kept={len(tx)} excluded={len(ex)}")
    pdf_sample = Path("data/sample_statement.pdf")
    if pdf_sample.exists():
        ptx, pex = parse_pdf(pdf_sample)
        print(f"PDF: kept={len(ptx)} excluded={len(pex)} balance_breaks={check_balance_continuity(ptx)}")
