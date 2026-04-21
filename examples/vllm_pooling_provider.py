"""
Build a vLLM pooling request for a supported multimodal plugin.
"""

from colsearch import DEFAULT_MULTIMODAL_MODEL_SPEC, VllmPoolingProvider


def main() -> None:
    spec = DEFAULT_MULTIMODAL_MODEL_SPEC
    provider = VllmPoolingProvider("http://127.0.0.1:8200", spec.model_id)
    payload = provider.build_payload(
        [{"text": "find the invoice total", "image": "/path/to/page.png"}],
        extra_kwargs={"pooling_task": spec.pooling_task},
    )
    print(payload)


if __name__ == "__main__":
    main()
