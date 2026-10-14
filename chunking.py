def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def chunk_by_paragraph(text):
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


with open("clinic_info.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Total characters in file:", len(text))

chunks = chunk_by_paragraph(text)

print("Number of chunks:", len(chunks))
print()

for i, chunk in enumerate(chunks, start=1):
    print(f"===== Chunk {i} ({len(chunk)} characters) =====")
    print(chunk)
    print()