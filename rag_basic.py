import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# 1. The clinic's "documents" (our knowledge base)
documents = [
    "Smile Dental Clinic is open from Saturday to Thursday, 10 AM to 9 PM. It is closed on Friday.",
    "A teeth cleaning session costs 500 EGP. Teeth whitening costs 2500 EGP.",
    "A tooth extraction costs 800 EGP. A dental implant costs 15000 EGP.",
    "The clinic is located at 15 Tahrir Street, Dokki, Giza.",
    "To book an appointment, call 01000000000 or send a WhatsApp message.",
    "Dr. Ahmed Samir specializes in implants. Dr. Mai Hassan specializes in children's dentistry.",
]


# 2. Retrieval: find the documents that share the most words with the question
def retrieve(question, docs, top_k=2):
    question_words = set(question.lower().split())
    scores = []
    for doc in docs:
        doc_words = set(doc.lower().split())
        score = len(question_words & doc_words)
        scores.append((score, doc))
    scores.sort(reverse=True)
    return [doc for score, doc in scores[:top_k]]


question = input("Ask about the clinic: ")
relevant_docs = retrieve(question, documents)

print("\n--- Retrieved documents ---")
for doc in relevant_docs:
    print("-", doc)

# 3. Augmentation: put the documents inside the prompt
context = "\n".join(relevant_docs)

prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}"""

# 4. Generation: the model answers from the context
response = client.chat.completions.create(
    model="deepseek/deepseek-v4.1-flash",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    max_tokens=1000,
)

print("\n--- Answer ---")
print(response.choices[0].message.content)