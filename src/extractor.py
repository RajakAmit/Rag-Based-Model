import pdfplumber


def extract_text_and_tables(pdf_path):
    """Extract text and tables row-by-row from PDF."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Text
            text = page.extract_text()
            if text:
                chunks.append(f"[Page {page_num}] {text}")

            # Tables
            tables = page.extract_tables()
            for t_index, table in enumerate(tables):
                for row in table:
                    row_text = " | ".join([cell if cell else "" for cell in row])
                    chunks.append(f"[Page {page_num} | Table {t_index}] {row_text}")
    return chunks

