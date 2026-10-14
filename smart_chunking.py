from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader("clinic_info.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    separators=["\n\n", ". ", " ", ""],
    keep_separator="end",
)


page_number = 4
page_text = reader.pages[page_number - 1].extract_text().replace("\n", " ").replace("Dr. ", "Dr ")
chunks = splitter.split_text(page_text)
print(f"Page {page_number} has {len(page_text)} characters")

print(f"Split into {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks):
    print(f"===== Chunk {i + 1} ({len(chunk)} chars) =====")
    print(chunk)
    print()