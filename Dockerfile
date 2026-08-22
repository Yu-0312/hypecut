# Single-stage on purpose: ffmpeg is the bulk of the image either way, and a
# one-container deployment is what most self-hosters actually want.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HYPECUT_DATA_DIR=/data \
    HYPECUT_WORKERS=1 \
    HYPECUT_MAX_UPLOAD_MB=4096

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir ".[web]"

# Run as a non-root user; /data is the only writable path needed at runtime.
RUN useradd --create-home --uid 10001 hypecut \
 && mkdir -p /data && chown -R hypecut:hypecut /data /app
USER hypecut

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/meta').read()" || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "hypecut.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
