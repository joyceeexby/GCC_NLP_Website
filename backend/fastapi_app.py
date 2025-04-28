import os
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize FastAPI app
app = FastAPI()

# Create necessary folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files (optional)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (safe for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# Load FinBERT model
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Define request body for /predict
class TextRequest(BaseModel):
    text: str

FINBERT_MODEL_URL = "https://finbert-api-271962191974.us-central1.run.app"  # Your model's endpoint

@app.post("/predict")
async def analyze_sentiment(request: Request):
    try:
        body = await request.json()
        text = body  # frontend sends raw text

        print(f"Received text: {text}")  # Log the incoming text

        # Tokenization and prediction
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            labels = ["negative", "neutral", "positive"]
            prediction = labels[torch.argmax(probs)]

        print(f"Prediction: {prediction}")  # Log the prediction

        return {
            "prediction": prediction
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
        
# /upload-dataset endpoint
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile):
    try:
        file_location = os.path.join("uploads", file.filename)
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        return {"message": f"Dataset '{file.filename}' uploaded successfully!", "path": file_location}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")

# /datasets endpoint
@app.get("/datasets")
async def list_datasets():
    try:
        files = os.listdir("uploads")
        return {"datasets": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

# / home page endpoint
@app.get("/")
async def home():
    return {
        "message": "Welcome to the GCC NLP Platform!",
        "endpoints": ["/predict", "/upload-dataset", "/datasets"],
    }