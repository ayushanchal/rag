import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def extract_qa_pairs(pdf_path: str):
    """
    Extract question-answer pairs from the Machine Learning A-Z Q&A PDF.
    """

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Combine all pages
    text = "\n".join(doc.page_content for doc in documents)

    # --------------------------------------------------
    # CLEANING
    # --------------------------------------------------

    # Remove page headers
    text = re.sub(
        r"Machine Learning A-Z Q&A",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove page numbers
    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove section titles like:
    # 2.1.1 Simple Linear Regression Intuition
    # 3.4 Kernel SVM
    text = re.sub(
        r"\n\d+(?:\.\d+)*\s+[A-Z][^\n]*",
        "\n",
        text,
    )

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize line breaks
    text = re.sub(r"\n{2,}", "\n", text)

    text = text.strip()

    # --------------------------------------------------
    # QUESTION DETECTION
    # --------------------------------------------------
    #
    # Capture:
    # Question ending with ?
    # followed by everything until next question.
    #
    pattern = re.compile(
        r"(?P<question>[^?\n]{10,}\?)\s*(?P<answer>.*?)(?=(?:[^?\n]{10,}\?)|$)",
        re.DOTALL,
    )

    qa_pairs = []

    for match in pattern.finditer(text):
        question = match.group("question").strip()
        answer = match.group("answer").strip()

        # Basic quality checks
        if len(question) < 10:
            continue

        if len(answer) < 5:
            continue

        answer = re.sub(r"\s+", " ", answer)
        qa_pairs.append(f"Question: {question}\nAnswer: {answer}")

    return qa_pairs



if __name__ == "__main__":

    pdf_path = "data\\Machine_Learning_A_Z_Q_A.pdf"

    qa_pairs = extract_qa_pairs(pdf_path)

    print(f"Found {len(qa_pairs)} Q&A pairs",'\n\n',qa_pairs)

