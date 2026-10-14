from pypdf import PdfReader

reader = PdfReader("clinic_info.pdf")
print(f"Number of pages: {len(reader.pages)}")

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"\n===== Page {i + 1} =====")
    print(text)