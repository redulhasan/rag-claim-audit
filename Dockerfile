FROM python:3.11-slim

WORKDIR /app

# cpu-only torch first, the default linux wheel pulls in several GB of cuda deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV RAG_LIGHTWEIGHT=1
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
