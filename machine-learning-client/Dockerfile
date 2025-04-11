FROM python:3.10-slim

# Install system dependencies for DeepFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

# Download DeepFace models during build (optional)
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet')"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]