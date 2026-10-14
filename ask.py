import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Open the saved database (no re-embedding!)
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("clinic")
print(f"Loaded {collection.count()} chunks from the database")


def retrieve(question, top_k=2):
    question_embedding = embedder.encode(question, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k,
    )
    docs = results["documents"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]
    return [
        (doc, 1 - distance / 2, meta["source"])
        for doc, distance, meta in zip(docs, distances, metas)
    ]


while True:
    question = input("\nAsk about the clinic (or 'exit'): ")
    if question.strip().lower() == "exit":
        break

    results = retrieve(question)

    print("\n--- Retrieved chunks ---")
    for chunk, score, source in results:
        first_line = chunk.split("\n")[0]
        print(f"{score:.2f}  [{first_line}]  (from {source})")

    context = "\n\n".join(chunk for chunk, score, source in results)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}"""

    response = client.chat.completions.create(
        model="deepseek/deepseek-v4.1-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )

    print("\n--- Answer ---")
    print(response.choices[0].message.content)