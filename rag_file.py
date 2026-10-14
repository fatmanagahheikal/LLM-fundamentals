import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_by_paragraph(text, min_length=50):
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_length]

# 1. Load the file
with open("clinic_info.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 2. Chunk it
chunks = chunk_by_paragraph(text)
print(f"Loaded {len(chunks)} chunks from clinic_info.txt")

# 3. Embed all chunks ONCE
chunk_embeddings = embedder.encode(chunks)


def retrieve(question, top_k=2):
    question_embedding = embedder.encode(question)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]


while True:
    question = input("\nAsk about the clinic (or 'exit'): ")
    if question == "exit":
        break

    results = retrieve(question)

    print("\n--- Retrieved chunks ---")
    for chunk, score in results:
        first_line = chunk.split("\n")[0]
        print(f"{score:.2f}  [{first_line}]")

    context = "\n\n".join(chunk for chunk, score in results)

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