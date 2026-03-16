FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# prebuilt dlib (fast install)
RUN pip install dlib-bin

COPY requirements.txt .

RUN pip install --prefer-binary -r requirements.txt

# install face-recognition without reinstalling dlib
RUN pip install face-recognition --no-deps

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]