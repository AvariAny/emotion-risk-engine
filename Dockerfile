FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instalar librerías básicas del sistema sin dependencias pesadas innecesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Instalar PyTorch CPU primero de forma aislada para evitar agotar la RAM
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Puerto configurable para Render ($PORT) y fallback a 8000 para Docker local
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]