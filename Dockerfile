FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build FAISS index at build time (requires GOOGLE_API_KEY as build arg)
# ARG GOOGLE_API_KEY
# RUN GOOGLE_API_KEY=$GOOGLE_API_KEY python build_index.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]