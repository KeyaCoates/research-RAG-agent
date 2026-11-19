FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY data/phak/corpus ./data/phak/corpus
COPY data/phak/index.npz ./data/phak/index.npz
RUN pip install --no-cache-dir ".[api,gcp]"
ENV PORT=8080
# GEMINI_API_KEY is injected at deploy time from Secret Manager (never baked into the image)
CMD exec uvicorn ragassist.api:app --host 0.0.0.0 --port $PORT
