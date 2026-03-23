"""Utilities for FAISS-based embedding indexing and nearest-neighbor retrieval.

This module provides :class:`FaissRetrievalIndex`, a small wrapper around FAISS
to support a practical retrieval workflow:

1. Create an index with a fixed embedding dimension and metric.
2. Add embeddings from in-memory arrays or from ``.npy``/``.pt`` files.
3. Search top-k nearest neighbors for one or many query vectors.
4. Save and reload both the FAISS index and the external ID mapping.

Supported capabilities
----------------------
- Index types: Flat and HNSW (via ``faiss.index_factory``)
- Metrics: inner product (``ip``) and Euclidean/L2 (``l2``)
- Input shapes: ``[D]`` (single vector) or ``[M, D]`` (batch)
- Optional L2 normalization applied consistently on add/search

Notes
-----
- IDs are managed in Python (``self.ids``) and aligned to FAISS row positions.
- For similarity search with inner product, set ``normalize=True`` if you want
    cosine-like behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import faiss
import torch


class FaissRetrievalIndex:
    """FAISS index wrapper with explicit external ID mapping.

    The class keeps FAISS vectors in insertion order and stores a parallel list
    of string IDs. Search returns both raw FAISS outputs (indices/distances) and
    mapped IDs for convenience.

    Parameters
    ----------
    embedding_dim:
        Expected dimensionality ``D`` of every vector inserted or queried.
    index_type:
        Index backend. Supported values are ``"flat"`` and ``"hnsw"``.
    metric:
        Distance/similarity metric. Supported values are ``"ip"`` and ``"l2"``.
    normalize:
        If ``True``, vectors are L2-normalized before add/search.
    hnsw_m:
        HNSW connectivity parameter used when ``index_type="hnsw"``.

    Raises
    ------
    ValueError
        If dimensions or configuration values are invalid.
    ImportError
        If required dependencies are missing.
    """

    _SUPPORTED_INDEX_TYPES = {"flat", "hnsw"}
    _SUPPORTED_METRICS = {"ip", "l2"}

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        metric: str = "ip",
        normalize: bool = False,
        hnsw_m: int = 32,
    ) -> None:
        if faiss is None:
            raise ImportError("faiss is required to use FaissRetrievalIndex")

        self.embedding_dim = int(embedding_dim)
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.normalize = normalize
        self.hnsw_m = int(hnsw_m)

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if self.index_type not in self._SUPPORTED_INDEX_TYPES:
            raise ValueError("index_type must be one of: flat, hnsw")
        if self.metric not in self._SUPPORTED_METRICS:
            raise ValueError("metric must be one of: ip, l2")

        self.index = self._build_index()
        self.ids: list[str] = []

    def _build_index(self) -> Any:
        """Build and return the configured FAISS index instance."""
        metric_type = (
            faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2
        )

        if self.index_type == "flat":
            factory = "Flat"
        else:
            factory = f"HNSW{self.hnsw_m},Flat"

        return faiss.index_factory(self.embedding_dim, factory, metric_type)

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Validate and format vectors for FAISS add/search operations.

        Parameters
        ----------
        vectors:
            Input vectors with shape ``[D]`` or ``[M, D]``.

        Returns
        -------
        np.ndarray
            Contiguous ``float32`` array with shape ``[M, D]``.

        Raises
        ------
        ValueError
            If shape rank is invalid or vector dimensionality mismatches
            ``self.embedding_dim``.
        """
        arr = np.asarray(vectors, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Embeddings must have shape [D] or [M, D]")
        if arr.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, got {arr.shape[1]}"
            )

        if self.normalize:
            arr = arr.copy()
            faiss.normalize_L2(arr)

        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _load_embedding_file(file_path: str | Path) -> np.ndarray:
        """Load embeddings from a supported file format.

        Parameters
        ----------
        file_path:
            Path to a ``.npy`` or ``.pt`` file containing an embedding array or
            tensor.

        Returns
        -------
        np.ndarray
            Loaded data converted to ``float32``.

        Raises
        ------
        ValueError
            If the file extension is unsupported.
        ImportError
            If ``.pt`` is requested and PyTorch is unavailable.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            data = np.load(path)
        elif suffix == ".pt":
            if torch is None:
                raise ImportError("PyTorch is required to load .pt files")
            tensor = torch.load(path, map_location="cpu")
            if hasattr(tensor, "detach"):
                tensor = tensor.detach().cpu().numpy()
            else:
                tensor = np.asarray(tensor)
            data = tensor
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .pt or .npy")

        return np.asarray(data, dtype=np.float32)

    def add(
        self,
        vectors: np.ndarray,
        ids: Iterable[str] | None = None,
    ) -> list[str]:
        """Append vectors to the FAISS index.

        Parameters
        ----------
        vectors:
            Array shaped ``[D]`` or ``[M, D]``.
        ids:
            Optional iterable of custom IDs. If omitted, sequential numeric IDs
            are generated based on current index size.

        Returns
        -------
        list[str]
            Final IDs inserted for the provided vectors.

        Raises
        ------
        ValueError
            If the number of IDs does not match the number of vectors.
        """
        arr = self._prepare_vectors(vectors)
        n = arr.shape[0]

        if ids is None:
            base = len(self.ids)
            final_ids = [str(base + i) for i in range(n)]
        else:
            final_ids = [str(i) for i in ids]
            if len(final_ids) != n:
                raise ValueError("Number of ids must match number of vectors")

        self.index.add(arr)
        self.ids.extend(final_ids)
        return final_ids

    def add_from_file(
        self, file_path: str | Path, base_id: str | None = None
    ) -> list[str]:
        """Append vectors loaded from a single file.

        Parameters
        ----------
        file_path:
            Path to a ``.pt`` or ``.npy`` file containing ``[D]`` or ``[M, D]``.
        base_id:
            Optional ID prefix. If omitted, the file stem is used.

        Returns
        -------
        list[str]
            Inserted IDs. For batched embeddings, IDs are expanded as
            ``base_id#0``, ``base_id#1``, ...

        Raises
        ------
        ValueError
            If the loaded data is not rank 1 or rank 2.
        """
        path = Path(file_path)
        vectors = self._load_embedding_file(path)
        arr = np.asarray(vectors)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(
                f"File {path} has invalid shape {arr.shape}; expected [D] or [M, D]"
            )

        if base_id is None:
            base_id = path.stem

        if arr.shape[0] == 1:
            ids = [str(base_id)]
        else:
            ids = [f"{base_id}#{i}" for i in range(arr.shape[0])]

        return self.add(arr, ids=ids)

    def add_many_files(self, file_paths: Iterable[str | Path]) -> list[str]:
        """Append embeddings from multiple files.

        Parameters
        ----------
        file_paths:
            Iterable of ``.pt``/``.npy`` paths.

        Returns
        -------
        list[str]
            Concatenation of all inserted IDs across files.
        """
        all_ids: list[str] = []
        for file_path in file_paths:
            added = self.add_from_file(file_path)
            all_ids.extend(added)
        return all_ids

    def search(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> dict[str, np.ndarray | list[list[str]]]:
        """Search nearest neighbors for one or many query vectors.

        Parameters
        ----------
        query_vectors:
            Query array shaped ``[D]`` or ``[Q, D]``.
        k:
            Number of neighbors to retrieve per query. Clipped to index size.

        Returns
        -------
        dict[str, np.ndarray | list[list[str]]]
            Dictionary with:
            - ``ids``: mapped string IDs per query result row
            - ``indices``: raw FAISS integer indices
            - ``distances``: raw FAISS distance/similarity scores

        Raises
        ------
        ValueError
            If index is empty or ``k`` is not positive.
        """
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Add vectors before search.")
        if k <= 0:
            raise ValueError("k must be > 0")

        q = self._prepare_vectors(query_vectors)
        k = min(k, self.index.ntotal)

        distances, indices = self.index.search(q, k)

        mapped_ids: list[list[str]] = []
        for row in indices:
            mapped_ids.append([self.ids[int(idx)] if idx >= 0 else "" for idx in row])

        return {
            "ids": mapped_ids,
            "indices": indices,
            "distances": distances,
        }

    def search_from_file(
        self, file_path: str | Path, k: int = 10
    ) -> dict[str, np.ndarray | list[list[str]]]:
        """Search using query vectors loaded from a file.

        Parameters
        ----------
        file_path:
            Path to a ``.pt`` or ``.npy`` query embedding file.
        k:
            Number of neighbors to retrieve per query.

        Returns
        -------
        dict[str, np.ndarray | list[list[str]]]
            Same payload format as :meth:`search`.
        """
        query = self._load_embedding_file(file_path)
        return self.search(query, k=k)

    def save(self, index_path: str | Path, mapping_path: str | Path) -> None:
        """Persist FAISS index and metadata/ID mapping to disk.

        Parameters
        ----------
        index_path:
            Destination path for the binary FAISS index.
        mapping_path:
            Destination path for JSON metadata and ordered ID list.
        """
        index_path = Path(index_path)
        mapping_path = Path(mapping_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        payload = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize": self.normalize,
            "hnsw_m": self.hnsw_m,
            "ids": self.ids,
        }
        mapping_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, index_path: str | Path, mapping_path: str | Path
    ) -> "FaissRetrievalIndex":
        """Load an index previously saved with :meth:`save`.

        Parameters
        ----------
        index_path:
            Path to the saved FAISS binary index.
        mapping_path:
            Path to the saved JSON metadata and ID mapping.

        Returns
        -------
        FaissRetrievalIndex
            Reconstructed index object with IDs restored.

        Raises
        ------
        ValueError
            If loaded FAISS vector count does not match number of stored IDs.
        """
        mapping_path = Path(mapping_path)
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))

        obj = cls(
            embedding_dim=int(payload["embedding_dim"]),
            index_type=str(payload["index_type"]),
            metric=str(payload["metric"]),
            normalize=bool(payload.get("normalize", False)),
            hnsw_m=int(payload.get("hnsw_m", 32)),
        )
        obj.index = faiss.read_index(str(index_path))
        obj.ids = [str(i) for i in payload["ids"]]

        if obj.index.ntotal != len(obj.ids):
            raise ValueError(
                "Index/vector count mismatch: "
                f"index.ntotal={obj.index.ntotal}, ids={len(obj.ids)}"
            )

        return obj
