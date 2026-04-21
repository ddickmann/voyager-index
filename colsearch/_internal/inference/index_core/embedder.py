"""
ColBERT Embedder - API Gateway Integration

Wraps the Latence client for ColBERT embeddings.
All requests route through the API gateway for tracking and billing.

This ensures that search SDK usage is properly metered and
the user's API key is validated.
"""

import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ColBERTEmbedder:
    """
    ColBERT embedding client that routes through Latence API gateway.

    All embedding requests are tracked for usage metering and billing.

    Example:
        >>> from latence import Latence
        >>> from latence.search.core import ColBERTEmbedder
        >>>
        >>> client = Latence(api_key="...")
        >>> embedder = ColBERTEmbedder(client)
        >>>
        >>> # Embed query
        >>> query_emb = await embedder.embed_query("What is machine learning?")
        >>>
        >>> # Embed documents in batches
        >>> doc_embs = await embedder.embed_documents(texts, batch_size=32)
    """

    def __init__(
        self,
        client: Any,
        default_dimension: int = 128,
        batch_size: int = 32,
    ):
        """
        Initialize ColBERT embedder.

        Args:
            client: Latence or AsyncLatence client
            default_dimension: Default embedding dimension
            batch_size: Default batch size for document embedding
        """
        self.client = client
        self.default_dimension = default_dimension
        self.batch_size = batch_size

        # Detect sync vs async client
        self._is_async = self._detect_async_client(client)

        logger.info(f"Initialized ColBERTEmbedder (async={self._is_async})")

    def _detect_async_client(self, client: Any) -> bool:
        """Detect if client is async."""
        # Check for AsyncLatence
        if hasattr(client, '__class__'):
            class_name = client.__class__.__name__
            if 'Async' in class_name:
                return True

        # Check for async methods
        if hasattr(client, 'embedding') and hasattr(client.embedding, '_async'):
            return False  # Sync client with async internals

        return False

    async def embed_query(
        self,
        query: str,
        dimension: Optional[int] = None,
    ) -> np.ndarray:
        """
        Embed a query string via API gateway.

        Args:
            query: Query text
            dimension: Embedding dimension (uses default if None)

        Returns:
            Query embedding as numpy array
        """
        dimension = dimension or self.default_dimension

        try:
            if self._is_async:
                result = await self.client.embedding.create(
                    text=query,
                    dimension=dimension,
                    encoding_format="float",
                )
            else:
                # Sync client - call directly (it handles async internally)
                result = self.client.embedding.create(
                    text=query,
                    dimension=dimension,
                    encoding_format="float",
                )

            # Extract embedding from response
            embedding = self._extract_embedding(result)

            logger.debug(f"Embedded query: {len(query)} chars -> {len(embedding)} dims")
            return embedding

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise

    async def embed_documents(
        self,
        documents: List[str],
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        """
        Embed documents in batches via API gateway.

        Args:
            documents: List of document texts
            dimension: Embedding dimension
            batch_size: Batch size (uses default if None)
            show_progress: Whether to show progress bar

        Returns:
            List of document embeddings
        """
        dimension = dimension or self.default_dimension
        batch_size = batch_size or self.batch_size

        embeddings = []
        total_docs = len(documents)

        # Optional progress tracking
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, total_docs, batch_size), desc="Embedding")
            except ImportError:
                iterator = range(0, total_docs, batch_size)
        else:
            iterator = range(0, total_docs, batch_size)

        for i in iterator:
            batch = documents[i:i + batch_size]
            batch_embeddings = await self._embed_batch(batch, dimension)
            embeddings.extend(batch_embeddings)

        logger.info(f"Embedded {len(embeddings)} documents")
        return embeddings

    async def _embed_batch(
        self,
        texts: List[str],
        dimension: int,
    ) -> List[np.ndarray]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts
            dimension: Embedding dimension

        Returns:
            List of embeddings
        """
        embeddings = []

        for text in texts:
            try:
                if self._is_async:
                    result = await self.client.embedding.create(
                        text=text,
                        dimension=dimension,
                        encoding_format="float",
                    )
                else:
                    result = self.client.embedding.create(
                        text=text,
                        dimension=dimension,
                        encoding_format="float",
                    )

                embedding = self._extract_embedding(result)
                embeddings.append(embedding)

            except Exception as e:
                logger.warning(f"Failed to embed text: {e}")
                # Return zero embedding as fallback
                embeddings.append(np.zeros(dimension, dtype=np.float32))

        return embeddings

    def _extract_embedding(self, result: Any) -> np.ndarray:
        """
        Extract embedding array from API response.

        Args:
            result: API response

        Returns:
            Numpy array of embedding
        """
        # Handle different response formats
        if isinstance(result, dict):
            if 'embedding' in result:
                emb = result['embedding']
            elif 'data' in result:
                data = result['data']
                if isinstance(data, list) and len(data) > 0:
                    emb = data[0].get('embedding', data[0])
                else:
                    emb = data
            else:
                emb = result
        elif hasattr(result, 'embedding'):
            emb = result.embedding
        elif hasattr(result, 'data'):
            emb = result.data[0].embedding
        else:
            emb = result

        # Convert to numpy
        if isinstance(emb, np.ndarray):
            return emb.astype(np.float32)
        elif isinstance(emb, list):
            return np.array(emb, dtype=np.float32)
        elif hasattr(emb, 'numpy'):
            return emb.numpy().astype(np.float32)
        else:
            return np.array(emb, dtype=np.float32)

    def embed_query_sync(
        self,
        query: str,
        dimension: Optional[int] = None,
    ) -> np.ndarray:
        """
        Synchronous version of embed_query.

        Args:
            query: Query text
            dimension: Embedding dimension

        Returns:
            Query embedding
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, need to use different approach
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.embed_query(query, dimension)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.embed_query(query, dimension))
        except RuntimeError:
            return asyncio.run(self.embed_query(query, dimension))

    def embed_documents_sync(
        self,
        documents: List[str],
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Synchronous version of embed_documents.

        Args:
            documents: List of documents
            dimension: Embedding dimension
            batch_size: Batch size

        Returns:
            List of embeddings
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.embed_documents(documents, dimension, batch_size)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.embed_documents(documents, dimension, batch_size)
                )
        except RuntimeError:
            return asyncio.run(self.embed_documents(documents, dimension, batch_size))


class ColBERTTokenEmbedder(ColBERTEmbedder):
    """
    ColBERT token-level embedder for multi-vector representations.

    Returns per-token embeddings instead of single vectors,
    enabling MaxSim scoring.
    """

    async def embed_query_tokens(
        self,
        query: str,
        max_tokens: int = 32,
    ) -> np.ndarray:
        """
        Embed query and return per-token embeddings.

        Args:
            query: Query text
            max_tokens: Maximum query tokens

        Returns:
            Token embeddings (num_tokens, dimension)
        """
        # For now, return single embedding expanded to token format
        # TODO: Integrate with actual ColBERT token embedding endpoint
        single_emb = await self.embed_query(query)

        # Expand to token format (placeholder)
        return single_emb.reshape(1, -1)

    async def embed_document_tokens(
        self,
        document: str,
        max_tokens: int = 512,
    ) -> np.ndarray:
        """
        Embed document and return per-token embeddings.

        Args:
            document: Document text
            max_tokens: Maximum document tokens

        Returns:
            Token embeddings (num_tokens, dimension)
        """
        # For now, return single embedding expanded to token format
        # TODO: Integrate with actual ColBERT token embedding endpoint
        single_emb = await self.embed_query(document)

        return single_emb.reshape(1, -1)

