from chunker import Chunker
from retriever import Retriever
from generator import Generator
from utils import load_documents


# Create Chunker
chunker = Chunker(
    chunk_size=100,
    overlap=20
)

# Load all documents
chunks = load_documents(
    "data",
    chunker
)

# Build Retriever
retriever = Retriever()

retriever.build_index(chunks)

# Ask User
query = input("Ask a question: ")


# Retrieve Relevant Chunks
results = retriever.search(query)


# Threshold Check
THRESHOLD = 0.30

if not results or results[0]["score"] < THRESHOLD:
    print("I don't know based on the provided documents.")

else:

    generator = Generator(
        api_key="YOUR_GROQ_API_KEY"
    )

    response = generator.generate(
        query,
        results
    )

    print("\nAnswer:\n")
    print(response)
