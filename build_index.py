import chromadb
from sentence_transformers import SentenceTransformer


def chunk_by_paragraph(text, min_length=50):
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_length]


# 1. Read and chunk the file
with open("clinic_info.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = chunk_by_paragraph(text)
print(f"Created {len(chunks)} chunks")

# 2. Convert chunks to vectors
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, normalize_embeddings=True)

# 3. Open (or create) the database folder
db = chromadb.PersistentClient(path="chroma_db")

# 4. Start fresh: delete the old collection if it exists
try:
    db.delete_collection("clinic")
except Exception:
    pass

collection = db.create_collection(name="clinic")

# 5. Save everything
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=[{"source": "clinic_info.txt"} for _ in chunks],
)

print(f"Saved {collection.count()} chunks to the 'chroma_db' folder")