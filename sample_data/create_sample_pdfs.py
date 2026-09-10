"""
Helper script to generate sample PDF files for manual UI testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import extract_text_from_pdf


def text_to_pdf_bytes(text: str) -> bytes:
    # Normalize unicode to standard ASCII
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    stream_content = "BT /F1 10 Tf 40 750 Td 13 TL "
    for line in lines[:45]:
        safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_content += f"({safe_line}) ' "
    stream_content += "ET"

    stream_bytes = stream_content.encode("latin1", errors="replace")

    pdf_str = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length {len(stream_bytes)} >>
stream
{stream_content}
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000117 00000 n 
0000000234 00000 n 
0000000300 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
380
%%EOF"""
    return pdf_str.encode("latin1", errors="replace")


def generate_sample_pdfs():
    base = Path(__file__).resolve().parent

    for txt_file in base.glob("*.txt"):
        pdf_file = txt_file.with_suffix(".pdf")
        text = txt_file.read_text(encoding="utf-8")
        pdf_bytes = text_to_pdf_bytes(text)
        pdf_file.write_bytes(pdf_bytes)
        
        # Verify extraction
        extracted, msg = extract_text_from_pdf(pdf_file)
        print(f"Generated: {pdf_file.name} | Status: {msg} | Characters: {len(extracted)}")


if __name__ == "__main__":
    generate_sample_pdfs()
