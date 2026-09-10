"""Build and load the RAG evidence corpus from pico_sections_icd11.json."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.io import read_etd_file

_EMPTY_EVIDENCE = re.compile(r"^react-empty:\s*\d+\s*$", re.IGNORECASE)
CORPUS_VERSION = "1"


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    text: str
    retrieval_text: str
    source_file: str
    pico_question: str
    criterion: str
    subquestion: str
    section_index: int
    chunk_index: int
    population: str
    intervention: str
    comparison: str
    main_outcomes: str


def evidence_to_text(evidence: Any) -> str:
    if evidence is None:
        return ""
    if isinstance(evidence, str):
        return evidence.strip()
    if isinstance(evidence, dict):
        parts: List[str] = []
        text = evidence.get("text")
        if text:
            parts.append(str(text).strip())
        for table in evidence.get("tables") or []:
            if not isinstance(table, dict):
                continue
            headers = table.get("headers") or []
            if headers:
                parts.append(" | ".join(str(h) for h in headers))
            for row in (table.get("rows") or [])[:8]:
                if isinstance(row, (list, tuple)):
                    parts.append(" | ".join(str(cell) for cell in row))
                else:
                    parts.append(str(row))
        return "\n".join(parts).strip()
    return str(evidence).strip()


def _is_usable_evidence(text: str) -> bool:
    if not text:
        return False
    if _EMPTY_EVIDENCE.match(text):
        return False
    return len(text) >= 40


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            start = 0
            while start < len(para):
                end = min(start + max_chars, len(para))
                chunks.append(para[start:end].strip())
                if end >= len(para):
                    break
                start = end - overlap_chars
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            flush()
            current = para
    flush()
    return chunks


def build_retrieval_text(
    *,
    pico: Dict[str, Any],
    section: Dict[str, Any],
    evidence_chunk: str,
) -> str:
    """Context prefix + evidence body, used for embedding / retrieval."""
    return "\n".join(
        [
            f"PICO question: {pico.get('Question', '')}",
            f"Population: {pico.get('Population', '')}",
            f"Intervention: {pico.get('Intervention', '')}",
            f"Comparison: {pico.get('Comparison', '')}",
            f"Outcomes: {pico.get('Main outcomes', '')}",
            f"EtD criterion: {section.get('criterion', '')}",
            f"EtD subquestion: {section.get('question', '')}",
            "",
            "Research evidence:",
            evidence_chunk,
        ]
    )


def build_corpus_from_dataset(
    dataset_path: Path,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[EvidenceChunk]:
    """
    Extract research_evidence from each PICO item's sections (by criterion),
    chunk for retrieval, and attach PICO / subquestion context.
    """
    etd_data = read_etd_file(dataset_path)
    chunks: List[EvidenceChunk] = []

    for item in etd_data:
        source_file = item["source_file"]
        source_stem = source_file.split(".")[0]
        pico = item["pico"]

        for section_index, section in enumerate(item["sections"]):
            raw = evidence_to_text(section.get("research_evidence"))
            if not _is_usable_evidence(raw):
                continue

            criterion = section.get("criterion", "")
            for chunk_index, piece in enumerate(
                chunk_text(
                    raw,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            ):
                chunk_id = f"{source_stem}::{section_index}::{chunk_index}"
                chunks.append(
                    EvidenceChunk(
                        chunk_id=chunk_id,
                        text=piece,
                        retrieval_text=build_retrieval_text(
                            pico=pico,
                            section=section,
                            evidence_chunk=piece,
                        ),
                        source_file=source_file,
                        pico_question=pico.get("Question", ""),
                        criterion=criterion,
                        subquestion=section.get("question", ""),
                        section_index=section_index,
                        chunk_index=chunk_index,
                        population=pico.get("Population", ""),
                        intervention=pico.get("Intervention", ""),
                        comparison=pico.get("Comparison", ""),
                        main_outcomes=pico.get("Main outcomes", ""),
                    )
                )
    return chunks


def save_corpus(chunks: List[EvidenceChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(c) for c in chunks]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_corpus(path: Path) -> List[EvidenceChunk]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    chunks: List[EvidenceChunk] = []
    for row in payload:
        if "retrieval_text" not in row:
            row["retrieval_text"] = row.get("text", "")
        chunks.append(EvidenceChunk(**row))
    return chunks


def build_corpus_stats(
    etd_data: List[Dict[str, Any]],
    chunks: List[EvidenceChunk],
) -> Dict[str, Any]:
    sections_with_evidence = 0
    sections_total = 0
    for item in etd_data:
        for section in item["sections"]:
            sections_total += 1
            raw = evidence_to_text(section.get("research_evidence"))
            if _is_usable_evidence(raw):
                sections_with_evidence += 1

    by_criterion: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for c in chunks:
        by_criterion[c.criterion] = by_criterion.get(c.criterion, 0) + 1
        stem = c.source_file.split(".")[0]
        by_source[stem] = by_source.get(stem, 0) + 1

    return {
        "corpus_version": CORPUS_VERSION,
        "pico_items": len(etd_data),
        "sections_total": sections_total,
        "sections_with_evidence": sections_with_evidence,
        "retrieval_chunks": len(chunks),
        "unique_pico_sources": len(by_source),
        "chunks_by_criterion": dict(sorted(by_criterion.items())),
    }


def build_and_save_corpus(
    dataset_path: Path,
    corpus_path: Path,
    meta_path: Optional[Path] = None,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[EvidenceChunk]:
    etd_data = read_etd_file(dataset_path)
    chunks = build_corpus_from_dataset(
        dataset_path,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    if not chunks:
        raise RuntimeError(
            f"No usable research_evidence found in {dataset_path}"
        )

    save_corpus(chunks, corpus_path)

    meta = build_corpus_stats(etd_data, chunks)
    meta["dataset_path"] = str(dataset_path.resolve())
    meta["corpus_path"] = str(corpus_path.resolve())
    meta["max_chars"] = max_chars
    meta["overlap_chars"] = overlap_chars

    meta_out = meta_path or corpus_path.parent / "meta.json"
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return chunks


def require_corpus(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"RAG corpus not found at {path}. "
            f"Run: python src/rag/build_corpus.py"
        )
    return path


def format_retrieved_evidence(
    hits: List[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    for rank, hit in enumerate(hits, start=1):
        meta = []
        if hit.get("source_file"):
            meta.append(f"source={hit['source_file']}")
        if hit.get("criterion"):
            meta.append(f"criterion={hit['criterion']}")
        if hit.get("score") is not None:
            meta.append(f"score={hit['score']:.4f}")
        header = f"[{rank}] ({', '.join(meta)})" if meta else f"[{rank}]"
        parts.append(f"{header}\n{hit['text']}")
    return "\n\n".join(parts)


def texts_for_embedding(chunks: List[EvidenceChunk]) -> List[str]:
    return [c.retrieval_text or c.text for c in chunks]
