# ---------- Stage 1: builder ----------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# build tools for compiling wheels (numpy etc.) — builder stage only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv settings that shrink and speed the build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Install dependencies only (cached unless deps change)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy source + model, install the project itself
COPY . .
RUN uv sync --frozen --no-dev

# ---------- Stage 2: final runtime ----------
FROM python:3.13-slim AS runtime

# Create the user FIRST, before copying anything
RUN useradd --create-home appuser

WORKDIR /app

# Copy with ownership set during the copy — no duplicate layer, no chown -R
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/models /app/models
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/README.md /app/

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "player_churn_model.api:app", "--host", "0.0.0.0", "--port", "8000"]