import os

def load_documents(folder_path, chunker):

    all_chunks = []

    for filename in os.listdir(folder_path):

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(folder_path, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = chunker.chunk_text(
            text,
            filename
        )

        all_chunks.extend(chunks)

    return all_chunks
