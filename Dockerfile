# ---------- Stage 1: build the Python environment ----------
FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Compilers are only needed here (some wheels, e.g. chromadb deps, may build from source)
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        build-essential \
#    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt


# ---------- Stage 2: slim runtime image ----------
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# Run as a non-root user
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser . .

# Chroma vector store is written to .vectordb/ - keep it writable (mount a volume here)
RUN mkdir -p /app/.vectordb && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Uses Python (slim has no curl). /docs is the FastAPI docs page.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/docs' % os.environ.get('PORT','8000'))" || exit 1

# Check your Makefile's `serve` target; change this line if it differs
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
