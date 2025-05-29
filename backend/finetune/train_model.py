import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Paths
REPORTS_DIR = "reports"
TEXTS_DIR = "extracted_texts"
MODEL_DIR = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

# Ensure text output folder exists
os.makedirs(TEXTS_DIR, exist_ok=True)

# 1. OCR Arabic PDFs
def ocr_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='ara') + "\n"
    return text

# 2. Extract and clean all PDFs
def extract_all_pdfs():
    texts = []
    for root, _, files in os.walk(REPORTS_DIR):
        for filename in files:
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(root, filename)
                print(f"🔍 Processing {pdf_path}...")
                raw_text = ocr_pdf(pdf_path)
                clean_text = raw_text.replace("\n", " ").strip()

                if len(clean_text) > 100:
                    texts.append({"text": clean_text})
                    print(f"Extracted {len(clean_text)} characters.")
                else:
                    print(f"Skipped {filename} (too short after OCR)")

                # Optional: save cleaned text
                relative_path = os.path.relpath(pdf_path, REPORTS_DIR)
                txt_filename = relative_path.replace(".pdf", ".txt").replace("/", "_")
                txt_path = os.path.join(TEXTS_DIR, txt_filename)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)
    print(f"Total documents extracted: {len(texts)}")
    return texts

# 3. Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)

# 4. Prepare dataset
def prepare_dataset(texts):
    dataset = Dataset.from_list(texts)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

# 5. Fine-tune
def train_model(dataset):
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        save_strategy="epoch",
        logging_steps=10,
        overwrite_output_dir=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")

if __name__ == "__main__":
    texts = extract_all_pdfs()
    dataset = prepare_dataset(texts)
    train_model(dataset)
    print("Fine-tuning complete. Model saved to ./finetuned_model")