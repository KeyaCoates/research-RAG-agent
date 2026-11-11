# research-RAG-agent
Agentic RAG assistant over papers + structured metadata: hybrid retrieval, multi-agent orchestration (ADK), eval-gated CI/CD via Azure DevOps, deployed on GCP Cloud Run

## Results (TBA)
| Pipeline | recall@5 | MRR | faithfulness | $/request | p95 latency |
|---|---|---|---|---|---|
| BM25 | | | | | |
| Dense | | | | | |
| Hybrid (RRF) | | | | | |
| + Reranker | | | | | |
| + Multi-agent (ADK) | | | | | |

## Quickstart (offline, no cloud needed)
```bash
pip install -e ".[dev,api]"
pytest -q
python -m ragassist.eval.run_eval --min-recall 0.6
uvicorn ragassist.api:app --port 8080
```

## Layout
- `src/ragassist/ingestion` - structure-aware, parent-child chunking, content hashing for incremental ingest
- `src/ragassist/retrieval` - BM25 + dense + RRF fusion
- `src/ragassist/pipeline.py` - baseline single-pass RAG
- `src/ragassist/telemetry.py` - spans, tokens, tokens/sec, TTFT, cost per request
- `src/ragassist/eval` - metrics and gate-able eval runner
- `infra/terraform` - GCP infra; `azure-pipelines.yml` - CI/CD; `docs/azure-devops.md` - setup

## Roadmap
- [ ] Real corpus ingestion (GCS -> parse -> embed -> Vertex Vector Search / pgvector)
- [ ] Vertex embeddings + Gemini generation (`LLM_BACKEND=vertex`)
- [ ] Reranker, then text-to-SQL path for structured data
- [ ] ADK multi-agent: router -> retriever / SQL -> critic (self-reflection)
- [ ] 100+ item golden set, faithfulness eval, ablation table
- [ ] OTel tracing to Cloud Trace, dashboard, load tests, guardrails