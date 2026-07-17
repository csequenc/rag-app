import os

from dotenv import load_dotenv

from chunker import Chunker
from retriever import Retriever
from reranker import Reranker
from generator import Generator
from utils import load_documents


# Configuration
CHUNK_SIZE = 100
OVERLAP = 20
TOP_K = 5
THRESHOLD = 0.30

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")


# Initialize Components
chunker = Chunker(
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP
)

chunks = load_documents(
    "data",
    chunker
)

retriever = Retriever()
retriever.build_index(chunks)

# reranker = Reranker()

generator = Generator(
    api_key=API_KEY
)