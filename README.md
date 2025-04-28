# GCC_NLP_Website

🚀 A simple and interactive web application for performing sentiment analysis on text using the finBERT model.

## Project Structure

- **frontend/**  
  Contains the `index.html` file which serves as the user-facing web interface for entering text, uploading datasets, and visualizing sentiment results.

- **backend/**  
  Built with FastAPI. Handles:
  - Receiving text input from the frontend
  - Sending requests to the deployed finBERT model (hosted on Google Cloud)
  - Returning the sentiment prediction to the frontend

## Features

- 🌟 Real-time sentiment analysis for user-entered text
- 📄 Upload datasets for batch analysis (future feature)
- 📈 Dynamic pie chart visualization of sentiment results
- 🔥 Fully connected frontend and backend architecture
- ☁️ Deployed finBERT model hosted separately on Google Cloud

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Backend**: FastAPI, Python
- **Model**: finBERT (Financial Sentiment BERT model)
- **Deployment**: Google Cloud Run (for model API)

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/joyceeexby/GCC_NLP_Website.git
   cd GCC_NLP_Website
