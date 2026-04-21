"""Vendored pure-Python LEMUR (Learned Multi-Vector Retrieval).

Source: https://github.com/ejaasaari/lemur (arxiv 2601.21853)
The official package requires AVX-512 for a C++ MaxSim kernel.
This vendor copies only the pure-Python training loop and models,
which use PyTorch for MaxSim. The production scoring path uses
our own Triton MaxSim kernel.
"""
from .lemur import Lemur

__all__ = ["Lemur"]
