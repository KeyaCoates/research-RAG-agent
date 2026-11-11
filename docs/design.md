# Design doc (fill in as you decide)
- Problem and users
- Corpus: unstructured (arXiv PDFs -> markdown) + structured (citation metadata table)
- Decisions to record with tradeoffs: vector store (Vertex Vector Search vs pgvector), chunking, reranker, router model
- Eval plan: 100+ golden Qs; recall@k, MRR, faithfulness; baseline vs hybrid vs rerank vs agentic
- Cost/latency targets and measured results
