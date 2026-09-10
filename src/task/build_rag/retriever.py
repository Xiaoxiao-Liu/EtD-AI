"""Dense retrieval with BAAI/bge-m3 over a pre-built RAG corpus and index."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.task.build_rag.corpus import (
    EvidenceChunk,
    format_retrieved_evidence,
    load_corpus,
    require_corpus,
    texts_for_embedding,
)
from src.task.build_rag.paths import default_bge_model_dir, require_bge_model_dir

DEFAULT_TOP_K = 5


def build_rag_index(
    corpus_path: Path,
    index_dir: Path,
    *,
    model_dir: Path | None = None,
    batch_size: int = 12,
    force: bool = False,
) -> Path:
    """
    Encode all corpus chunks with BGE-M3 and save embeddings.npy + meta.json.
    Run once (or again after corpus / model changes).
    """
    corpus_path = require_corpus(corpus_path)
    resolved_model_dir = require_bge_model_dir(
        model_dir or default_bge_model_dir()
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "meta.json"
    model_path_key = str(resolved_model_dir.resolve())
    corpus_path_key = str(corpus_path.resolve())

    if not force and embeddings_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("model_path") == model_path_key
            and meta.get("corpus_path") == corpus_path_key
        ):
            chunks = load_corpus(corpus_path)
            embeddings = np.load(embeddings_path)
            if len(chunks) == len(embeddings):
                print(
                    f"Index up to date ({len(chunks)} vectors) at {index_dir}, skip."
                )
                return embeddings_path

    print(f"Encoding {corpus_path} with BGE-M3 ...")
    chunks = load_corpus(corpus_path)
    if not chunks:
        raise RuntimeError(f"Empty corpus at {corpus_path}")

    model = _load_model(resolved_model_dir)
    texts = texts_for_embedding(chunks)
    embeddings = _encode_dense(model, texts, batch_size=batch_size)
    embeddings = _normalize(embeddings)

    np.save(embeddings_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": model_path_key,
                "corpus_path": corpus_path_key,
                "num_chunks": len(chunks),
            },
            f,
            indent=2,
        )
    print(f"Saved {len(chunks)} vectors -> {embeddings_path}")
    return embeddings_path


def require_rag_index(
    corpus_path: Path,
    index_dir: Path,
    *,
    model_dir: Path | None = None,
) -> None:
    corpus_path = require_corpus(corpus_path)
    resolved_model_dir = require_bge_model_dir(
        model_dir or default_bge_model_dir()
    )
    embeddings_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "meta.json"
    if not embeddings_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"RAG vector index not found under {index_dir}. "
            f"Run: python src/rag/build_index.py"
        )
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("corpus_path") != str(corpus_path.resolve()):
        raise FileNotFoundError(
            f"Index corpus_path mismatch. Rebuild: "
            f"python src/rag/build_index.py --force"
        )
    if meta.get("model_path") != str(resolved_model_dir.resolve()):
        raise FileNotFoundError(
            f"Index model_path mismatch. Rebuild: "
            f"python src/rag/build_index.py --force"
        )
    chunks = load_corpus(corpus_path)
    embeddings = np.load(embeddings_path)
    if len(chunks) != len(embeddings):
        raise FileNotFoundError(
            f"Index size mismatch ({len(embeddings)} vs {len(chunks)} chunks). "
            f"Run: python src/rag/build_index.py --force"
        )


class BGERetriever:
    def __init__(
        self,
        chunks: List[EvidenceChunk],
        embeddings: np.ndarray,
        *,
        model_dir: Path | None = None,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self.chunks = chunks
        self.embeddings = embeddings
        self.model_dir = require_bge_model_dir(
            model_dir or default_bge_model_dir()
        )
        self._model = None
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        corpus_path: Path,
        index_dir: Path,
        *,
        model_dir: Path | None = None,
    ) -> "BGERetriever":
        """Load pre-built corpus + embeddings (no encoding)."""
        require_rag_index(corpus_path, index_dir, model_dir=model_dir)
        resolved_model_dir = require_bge_model_dir(
            model_dir or default_bge_model_dir()
        )
        chunks = load_corpus(corpus_path)
        embeddings = np.load(index_dir / "embeddings.npy")
        print(f"Loaded RAG index: {len(chunks)} chunks from {index_dir}")
        return cls(chunks, embeddings, model_dir=resolved_model_dir)

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self.model_dir)
        return self._model

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        exclude_source_file: Optional[str] = None,
        batch_size: int = 12,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            model = self._get_model()
            query_vec = _encode_dense(model, [query], batch_size=batch_size)[0]
        query_vec = _normalize(query_vec.reshape(1, -1))[0]

        scores = self.embeddings @ query_vec
        ranked_indices = np.argsort(scores)[::-1]

        hits: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            chunk = self.chunks[int(idx)]
            if (
                exclude_source_file
                and chunk.source_file == exclude_source_file
            ):
                continue
            hits.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "criterion": chunk.criterion,
                    "subquestion": chunk.subquestion,
                    "section_index": chunk.section_index,
                    "score": float(scores[int(idx)]),
                }
            )
            if len(hits) >= top_k:
                break
        return hits

    def retrieve_formatted(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        exclude_source_file: Optional[str] = None,
    ) -> tuple[str, List[Dict[str, Any]]]:
        hits = self.retrieve(
            query,
            top_k=top_k,
            exclude_source_file=exclude_source_file,
        )
        return format_retrieved_evidence(hits), hits


def _patch_flagembedding_transformers_compat() -> None:
    """FlagEmbedding 1.4.x passes ``dtype=`` to ``AutoModel.from_pretrained``.

    transformers >= 4.54 forwards that kwarg into ``XLMRobertaModel.__init__``,
  which does not accept it. Remap to the supported ``torch_dtype`` alias.
    """
    if getattr(_patch_flagembedding_transformers_compat, "_applied", False):
        return

    from transformers import AutoModel

    original = AutoModel.from_pretrained.__func__  # type: ignore[attr-defined]

    def _from_pretrained(cls, *args, **kwargs):
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return original(cls, *args, **kwargs)

    AutoModel.from_pretrained = classmethod(_from_pretrained)  # type: ignore[method-assign]
    _patch_flagembedding_transformers_compat._applied = True  # type: ignore[attr-defined]


def _load_model(model_dir: Path):
    model_dir = require_bge_model_dir(model_dir)
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise ImportError(
            "FlagEmbedding is required for BGE-M3 retrieval. "
            "Install with: pip install FlagEmbedding"
        ) from exc
    _patch_flagembedding_transformers_compat()
    return BGEM3FlagModel(str(model_dir), use_fp16=True)


def _encode_dense(model, texts: List[str], *, batch_size: int) -> np.ndarray:
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=8192,
    )
    if isinstance(output, dict):
        vecs = output["dense_vecs"]
    else:
        vecs = output
    return np.asarray(vecs, dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms
