FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY data/eval/corpus ./data/eval/corpus
RUN pip install --no-cache-dir ".[api,gcp,obs]"
ENV PORT=8080
CMD exec uvicorn ragassist.api:app --host 0.0.0.0 --port $PORT
