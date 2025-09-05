# Use a small Python base
FROM python:3.11-slim

# Prevent Python from writing pyc files / enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (CA certs for HTTPS to AWS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY app.py ./

# Create logs directory (your app writes here)
RUN mkdir -p /app/logs

# Expose Flask port
EXPOSE 8000

# Healthcheck: hit /health endpoint using python -c (no heredoc)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python","-c","import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else sys.exit(1)"]

# Run your app (keeps your background threads + trigger setup)
CMD ["python", "app.py"]
