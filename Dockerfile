FROM python:3.11-slim

WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# COPY api AS A PACKAGE (IMPORTANT FIX)
COPY api/ /app/api/

# make /app visible for imports
ENV PYTHONPATH=/app

# install dependencies
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# IMPORTANT: must use api.main
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]