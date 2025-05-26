import fitz  # PyMuPDF
from transformers import pipeline

# Load sentiment model
sa = pipeline("text-classification", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")

# Extract text from PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Test PDF file path (replace with your file)
pdf_path = "usb Annual report Arabic 2018 softproof.pdf"
arabic_text = extract_text(pdf_path)

# Break text into chunks (max 512 tokens per inference)
chunks = [arabic_text[i:i+512] for i in range(0, len(arabic_text), 512)]

# Run sentiment analysis
for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks
    print(f"\nChunk {i+1}:")
    print(sa(chunk))