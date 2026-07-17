FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 libffi-dev \
    fonts-dejavu-core libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# шрифты DejaVu — кириллица в PDF (ReportLab fallback)

ENV PORT=8080 TEST_MODE=1 DB_PATH=/data/deals.db KP_DIR=/data/kp
EXPOSE 8080
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT} app:app --timeout 120"]
