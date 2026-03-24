import pdfplumber


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from an uploaded PDF file.

    Args:
        pdf_file: Uploaded file object from Streamlit.

    Returns:
        Extracted text as a string.
    """
    extracted_text = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(page_text)
    except Exception as e:
        raise ValueError(f"Error reading PDF: {e}")

    return "\n".join(extracted_text).strip()