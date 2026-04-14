FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cargo \
    curl \
    rustc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE LICENSING.md THIRD_PARTY_NOTICES.md /app/
COPY voyager_index /app/voyager_index
COPY src /app/src

RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools maturin \
    && python -m pip wheel --no-deps --wheel-dir /wheels . \
    && python -m pip wheel --no-deps --wheel-dir /wheels ./src/kernels/knapsack_solver

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VOYAGER_INDEX_PATH=/data/voyager-index
ENV HOST=0.0.0.0
ENV PORT=8080
ENV WORKERS=4

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system voyager \
    && useradd --system --gid voyager --create-home --home-dir /home/voyager voyager \
    && mkdir -p /data/voyager-index \
    && chown -R voyager:voyager /data /home/voyager

COPY --from=builder /wheels /wheels

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir /wheels/latence_solver-*.whl \
    && VOYAGER_WHEEL=$(ls /wheels/voyager_index-*.whl) \
    && python -m pip install --no-cache-dir "voyager-index[server,preprocessing,shard] @ file://${VOYAGER_WHEEL}"

USER voyager

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" || exit 1

CMD ["voyager-index-server"]
