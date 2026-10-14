from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Which doctor treats kids?",
    "Dr. Mai Hassan specializes in children's dentistry.",
    "A teeth cleaning session costs 500 EGP.",
    "The clinic is closed on Friday.",
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)
print("First 5 numbers of sentence 1:", embeddings[0][:5])
print()

scores = util.cos_sim(embeddings[0], embeddings[1:])

print("Similarity with: 'Which doctor treats kids?'")
for sentence, score in zip(sentences[1:], scores[0]):
    print(f"{float(score):.2f}  {sentence}")