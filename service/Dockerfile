FROM python:3.10-slim

RUN apt-get update && apt-get install -y g++ gcc gdal-bin libgdal-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /app
COPY app.py /app/app.py
CMD ["python", "app.py"]
