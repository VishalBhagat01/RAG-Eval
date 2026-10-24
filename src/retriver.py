import os
from pathlib import Path
import re
import glob

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(str(DATA_DIR / "*.vtt")):
        lines = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        match = re.search(r"Session[ _]*(\d+)", path, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Could not find a session number in transcript path: {path}")
        session = match.group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def _embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    if provider == "ollama":
        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL")
            or os.getenv("LOCAL_MODEL_BASE_URL")
            or None,
        )
    if provider in {"huggingface", "hf"}:
        return HuggingFaceEmbeddings(
            model_name=os.getenv(
                "HUGGINGFACE_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        )
    raise ValueError(
        "Unsupported EMBEDDING_PROVIDER. Use 'huggingface' or 'ollama'."
    )


def load_store():
    embeddings = _embeddings()
    store = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    if store._collection.count() > 0:
        return store

    docs = load_transcripts()
    if not docs:
        raise FileNotFoundError(f"No .vtt transcripts found in {DATA_DIR}")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(DB_DIR),
    )


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")