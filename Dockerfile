FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app


COPY pyproject.toml .
RUN uv sync --group test

COPY . .

ENV PYTHONPATH=/app

CMD ["uv", "run", "pytest", "tests/", "-v"]