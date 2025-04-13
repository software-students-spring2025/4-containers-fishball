FROM python:3.10-slim

# Install system dependencies for DeepFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5 /root/.deepface/weights/retinaface.h5
ADD https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5 /root/.deepface/weights/gender_model_weights.h5
ADD https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5 /root/.deepface/weights/facial_expression_model_weights.h5
ADD https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5 /root/.deepface/weights/age_model_weights.h5  

COPY src/ .

# Download DeepFace models during build (optional)
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet')"

CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--timeout", "120", "app:app"]