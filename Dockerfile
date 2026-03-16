FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install prebuilt dlib (avoids long compile)
RUN pip install dlib-bin

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --prefer-binary -r requirements.txt

# Install face-recognition without installing dlib again
RUN pip install face-recognition --no-deps

# Copy project files
COPY . .

# Expose API port
EXPOSE 10000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]