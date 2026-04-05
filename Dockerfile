FROM python:3.11-slim

# Create a non-root user
RUN groupadd -r datara && useradd -r -g datara datara

WORKDIR /app

# System deps — curl is required for the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY datara_env/ /app/datara_env/
COPY openenv.yaml /app/
COPY inference.py /app/
COPY scripts/demo_agent.py /app/
COPY README.md /app/

# Set ownership
RUN chown -R datara:datara /app

USER datara

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "datara_env.server:app", "--host", "0.0.0.0", "--port", "8000"]