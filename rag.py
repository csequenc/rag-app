import os
from dotenv import load_dotenv

from chunker import Chunker
from retriever import Retriever
from reranker import Reranker
from generator import Generator
from utils import load_documents

CHUNK_SIZE = 100
OVERLAP = 20
TOP_K = 5

chunker = None
retriever = None
reranker = None
generator = None


def initialize_rag():
    global chunker, retriever, reranker, generator

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    chunker = Chunker(
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    chunks = load_documents("data", chunker)

    retriever = Retriever()
    retriever.build_index(chunks)

    reranker = Reranker()

    generator = Generator(api_key=api_key)
    

def run_rag(question):

    results = retriever.search(question, TOP_K)

    results = reranker.rerank(question, results)

    answer = generator.generate(question, results)

    retrieved_documents = list(
        {chunk["source"] for chunk in results}
    )

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": results,
        "retrieved_documents": retrieved_documents
    }