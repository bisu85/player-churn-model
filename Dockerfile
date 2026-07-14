# Start from a slim Python image matching your local version
FROM python:3.13-slim

# Bring in the uv binary from its official image (fast, no pip install needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) Install dependencies only — this layer is cached unless deps change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Copy the app code + trained model, then install the project itself
COPY . .
RUN uv sync --frozen --no-dev

# Put the virtualenv on PATH so we can call uvicorn directly
ENV PATH="/app/.venv/bin:$PATH"

# Document the port the app listens on
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "player_churn_model.api:app", "--host", "0.0.0.0", "--port", "8000"]