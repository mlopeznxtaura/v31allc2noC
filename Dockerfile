FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

COPY requirements.deploy.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir -r requirements.txt

COPY v31 ./v31
COPY server.py .
COPY static ./static

EXPOSE 8080
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
