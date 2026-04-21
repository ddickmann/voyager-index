from __future__ import annotations

import torch

from colsearch import fast_colbert_scores


def _reference_scores(
    query: torch.Tensor,
    documents: torch.Tensor,
    documents_mask: torch.Tensor,
) -> torch.Tensor:
    query = torch.nn.functional.normalize(query.to(torch.float32), p=2, dim=-1)
    documents = torch.nn.functional.normalize(documents.to(torch.float32), p=2, dim=-1)
    similarities = torch.einsum("ash,bth->abst", query, documents)
    similarities = similarities.masked_fill(~documents_mask.to(torch.bool)[None, :, None, :], float("-inf"))
    return torch.nan_to_num(similarities.max(dim=-1).values, neginf=0.0).sum(dim=-1)


def test_fast_colbert_scores_matches_masked_reference() -> None:
    query = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    documents = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ]
    )
    documents_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    scores = fast_colbert_scores(
        query,
        documents,
        documents_mask=documents_mask,
    )
    expected = _reference_scores(query, documents, documents_mask)

    assert torch.allclose(scores, expected, atol=1e-5)
    assert float(scores[0, 1]) > float(scores[0, 0])
