"""Default filesystem paths for the build_rag stage:
- Corpus / vector index under ``EtD/dataset/build_rag/``
- Local BGE-M3 model snapshot under ``EtD/model/``
"""

from pathlib import Path

ETD_ROOT = Path(__file__).resolve().parents[3]

BGE_M3_REPO_ID = "BAAI/bge-m3"
BGE_M3_DIR = ETD_ROOT / "model" / "bge-m3"

RAG_CORPUS_DIR = ETD_ROOT / "dataset" / "build_rag" / "corpus" / "output"
DEFAULT_CORPUS_PATH = RAG_CORPUS_DIR / "corpus.json"
DEFAULT_CORPUS_META_PATH = RAG_CORPUS_DIR / "meta.json"

RAG_INDEX_DIR = ETD_ROOT / "dataset" / "build_rag" / "index" / "output"


def default_corpus_path(etd_root: Path | None = None) -> Path:
    return (etd_root or ETD_ROOT) / "dataset" / "build_rag" / "corpus" / "output" / "corpus.json"


def default_corpus_meta_path(etd_root: Path | None = None) -> Path:
    return (etd_root or ETD_ROOT) / "dataset" / "build_rag" / "corpus" / "output" / "meta.json"


def default_index_dir(etd_root: Path | None = None) -> Path:
    return (etd_root or ETD_ROOT) / "dataset" / "build_rag" / "index" / "output" / "bge-m3"


def default_embeddings_path(etd_root: Path | None = None) -> Path:
    return default_index_dir(etd_root) / "embeddings.npy"


def default_index_meta_path(etd_root: Path | None = None) -> Path:
    return default_index_dir(etd_root) / "meta.json"


def default_bge_model_dir(etd_root: Path | None = None) -> Path:
    return (etd_root or ETD_ROOT) / "model" / "bge-m3"


def require_bge_model_dir(model_dir: Path | None = None) -> Path:
    path = model_dir or BGE_M3_DIR
    config = path / "config.json"
    if not config.is_file():
        raise FileNotFoundError(
            f"BGE-M3 model not found at {path}. "
            f"Run: python -m src.common.download_model"
        )
    return path
