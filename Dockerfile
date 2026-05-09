FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
