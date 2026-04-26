FROM python:3.12-slim

# Better logging + performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Poetry
RUN pip install --no-cache-dir poetry

WORKDIR /app

# Copy only dependency files first (caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies (production only)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# Copy the rest of your code
COPY . .

EXPOSE 8080

# ←←← This is the exact production command you need
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]