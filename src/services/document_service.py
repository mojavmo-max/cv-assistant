from  pathlib import Path
import pdfplumber
from docx import Document
import re
from collections import Counter
from services.llm_service import generate_structured_data

def process_document(path: Path) -> str:
    """Process a document file (PDF or DOCX) and return cleaned text content."""
    
    text = load_file(path)
    return generate_structured_data(text)

def load_file(path: Path) -> str:
    """Load a PDF file and return its content as a string."""
    
    text = ""
    extension = path.split('.')[-1].lower()
    if extension == "pdf":
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return clean_text(text)
    elif extension == "docx":
        doc = Document(path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return clean_text(text)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return text

def clean_text(text: str) -> str:
    """Clean the extracted text by removing extra whitespace and other noise."""

    #Remove page numbers
    text = re.sub(r'\bPage \d+\b', '', text)
    text = re.sub(r'-\s*\d+\s*-', '', text)

    #Remove weird symbols
    text = re.sub(r'[^\w\s.,!?()-]', ' ', text)

    #Remove repeated headers/footers
    lines = text.splitlines()
    line_counts = Counter(lines)
    lines = [line for line in lines if line_counts[line] < 3]

    text = '\n'.join(lines)

    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)  

    return text.strip()