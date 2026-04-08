FROM python:3.11-slim

LABEL maintainer="AI Hackathon Team"
LABEL description="AI Customer Support Training Environment (OpenEnv)"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Root-level files
COPY main.py         .
COPY inference.py    .
COPY index.html      .
COPY openenv.yaml    .
COPY README.md       .

# Environment subfolder (models, tasks, graders, environment, __init__)
COPY environment/    ./environment/

# Non-sensitive config only
ENV PORT=7860

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]