import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Load the embedding model (runs on your laptop)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Smile Dental Clinic is open from Saturday to Thursday, 10 AM to 9 PM. It is closed on Friday.",
    "A teeth cleaning session costs 500 EGP. Teeth whitening costs 2500 EGP.",
    "A tooth extraction costs 800 EGP. A dental implant costs 15000 EGP.",
    "The clinic is located at 15 Tahrir Street, Dokki, Giza.",
    "To book an appointment, call 01000000000 or send a WhatsApp message.",
    "Dr. Ahmed Samir specializes in implants. Dr. Mai Hassan specializes in children's dentistry.",
        "The clinic accepts Visa and Mastercard credit and debit cards, and cash.",
]

# Phase 1: convert all documents to vectors ONCE
doc_embeddings = embedder.encode(documents)


# Phase 2: for each question, find the closest documents
def retrieve(question, top_k=2):
    question_embedding = embedder.encode(question)
    scores = util.cos_sim(question_embedding, doc_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return [(documents[i], float(scores[i])) for i in top_indices]


while True:
    question = input("\nAsk about the clinic (or 'exit'): ")
    if question == "exit":
        break

    results = retrieve(question)

    print("\n--- Retrieved documents ---")
    for doc, score in results:
        print(f"{score:.2f}  {doc}")

    context = "\n".join(doc for doc, score in results)

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