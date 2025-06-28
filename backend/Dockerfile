# Dockerfile
FROM python:3.10-slim

# Optional: Add system dependencies if needed
RUN apt-get update && apt-get install -y gcc

# Set working directory
WORKDIR /app

# Copy source code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn torch transformers

# Run the app
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080"]