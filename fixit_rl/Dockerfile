FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-train on startup so dashboard has results immediately
RUN python train.py --episodes 2000 --eval-episodes 300 --seed 42

EXPOSE 8000

CMD ["python", "api/server.py"]
