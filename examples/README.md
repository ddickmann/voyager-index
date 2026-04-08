# Examples

These examples are ordered from first-run user flow to optional provider
integration.

For the full step-by-step feature walkthrough, pair these examples with
`docs/full_feature_cookbook.md`.

## Recommended Order

1. `examples/reference_api_happy_path.py`
2. `examples/reference_api_feature_tour.py`
3. `examples/reference_api_late_interaction.py`
4. `examples/reference_api_multimodal.py`
5. `examples/vllm_pooling_provider.py`

## What Each Example Covers

- `examples/reference_api_happy_path.py`: end-to-end OSS HTTP flow across `dense`, `late_interaction`, and `multimodal` collections, including filters and truthful notes about optional features
- `examples/reference_api_feature_tour.py`: advanced step-by-step feature tour with progress logging, optional JSON report output, CRUD coverage, BM25/hybrid search, solver-refinement checks, multimodal screening, and boundary checks
- `examples/reference_api_late_interaction.py`: the smallest possible late-interaction collection roundtrip with precomputed embeddings
- `examples/reference_api_multimodal.py`: the smallest possible multimodal collection roundtrip with precomputed patch embeddings
- `examples/vllm_pooling_provider.py`: building requests for an optional user-operated vLLM pooling endpoint

## Requirements

- `examples/reference_api_happy_path.py`
- `examples/reference_api_feature_tour.py`
- `examples/reference_api_late_interaction.py`
- `examples/reference_api_multimodal.py`

These require the reference server to be running:

```bash
pip install voyager-index[server]
voyager-index-server
python examples/reference_api_happy_path.py
python examples/reference_api_feature_tour.py --output-json feature-tour-report.json
```

Use `README.md` for the full install matrix and `http://127.0.0.1:8080/docs` for the live OpenAPI surface once the server is running.

- `examples/vllm_pooling_provider.py`

This does not require the server, but it does assume you understand the
multimodal provider seam documented in `MULTIMODAL_FOUNDATION.md`.
