from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import uvicorn

app = FastAPI()

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Define request structure
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(request: TextRequest):
    text = request.text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        labels = ["negative", "neutral", "positive"]
        prediction = labels[torch.argmax(probs)]
    return {"prediction": prediction}

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)