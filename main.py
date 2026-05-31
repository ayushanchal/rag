from email import message
import os
from langchain_classic.chains import RetrievalQA
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter, SentenceTransformersTokenTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from huggingface_hub import login, whoami
from dotenv import load_dotenv
import re
from sentence_transformers import CrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch


load_dotenv()
hf_token = os.getenv("HF_TOKEN")


if hf_token:
    login(hf_token)
    user_info = whoami()
    print("✅ Successfully logged in to Hugging Face")
    print(f"Username: {user_info['name']}")
else:
    print("❌ HF_TOKEN environment variable is not set")


def load_documents(data_path="data"):
    docs = []
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            documents = loader.load()
            documents = documents[4:]
            docs = "\n".join([doc.page_content for doc in documents])
            #docs.extend(documents)
    with open("document.txt", "w", encoding="utf-8") as file:
        file.write(str(docs))
    return docs

def split_documents(documents):
    '''#text_splitter = RecursiveCharacterTextSplitter(
    #text_splitter = SpacyTextSplitter(
    #text_splitter = SentenceTransformersTokenTextSplitter(
    #text_splitter = CharacterTextSplitter(
        separator="\n\n",
    )
    return text_splitter.split_documents(documents)'''
    pattern = r"(\n.*?\n.*?\?)\s*(.*?)(?=\n.*?\n.*?\?|$)"

    matches = re.findall(pattern, documents, re.DOTALL)
    
    #print('matches:\n', len(matches), '\n', type(matches), matches)
    
    chunks = []

    for question, answer in matches:

        chunk = f"""
Question:
{question.strip()}

Answer:
{answer.strip()}
"""

        chunks.append(chunk)
    #print('chunk size:\n', len(chunks), '\n', type(chunks), chunks)
    return chunks



def load_split(pdf_path: str):
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

def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cuda"
        }
    )


def create_vectorstore(chunks, embeddings):
    # Convert string chunks to Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def load_llm():

    model_name = "Qwen/Qwen3-4B"  # For RTX 3050 Ti 4GB

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3
    )

    return HuggingFacePipeline(pipeline=pipe)


def create_qa_chain(vectorstore, llm):
    # Step 1: Retrieve more documents using MMR
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 20,
            "fetch_k": 50
        }
    )

    # Step 2: Load reranker
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
        model_kwargs={
            "device": "cuda"
            }
    )

    # Step 3: Keep only the best 5 chunks
    compressor = CrossEncoderReranker(
        model=model,
        top_n=5
    )

    # Step 4: Wrap retriever with reranker
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # Step 5: Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    return qa

if __name__ == "__main__":
    print("Loading documents...")
    #documents = load_documents()

    print("Splitting documents...")
    #chunks = split_documents(documents)
    chunks = load_split("data/Machine_Learning.pdf")

    '''
    all= '\n'.join(f"chunk:\n{chunk}\n\n" for chunk in chunks)
    print(type(all))
    with open("chunks5.txt", "w", encoding="utf-8") as file:
       file.write(all)
    '''


    print("Creating embeddings...")
    embeddings = create_embeddings()
    

    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks, embeddings)
    
    print("Loading LLM...")
    llm = load_llm()

    print("Creating QA chain...")
    qa_chain = create_qa_chain(vectorstore, llm)

    print("\nSystem Ready! Ask questions (type 'exit' to quit)\n")

    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        result = qa_chain.invoke({"query": query})

        print("\nAnswer:\n", result['result'])

        '''print("\nSources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source"))'''

