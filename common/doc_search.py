"""轻量文档索引：一次加载、分段 BM25 检索，不依赖第三方搜索服务。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._+-]*|[\u4e00-\u9fff]+", re.I)
_SPACE_RE = re.compile(r"\s+")
_DROP_LINE_RE = re.compile(
    r"^\s*(?:\[End OCR\]|##\s*Page\s+\d+|Slide number:\s*\d+|目录)\s*$",
    re.I,
)


def normalize_text(text: str) -> str:
    """保留中英文与数字，去掉空白/标点，用于短语命中。"""
    return "".join(_TOKEN_RE.findall(text.lower()))


def tokenize(text: str) -> list[str]:
    """英文按词、中文按二元字切分；无需 jieba 也能匹配专业混合术语。"""
    tokens: list[str] = []
    for part in _TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
                if len(part) <= 6:
                    tokens.append(part)
        else:
            tokens.append(part)
    return tokens


def is_low_quality_snippet(snippet: str) -> bool:
    text = snippet.strip()
    if len(text) < 24:
        return True
    visible_lines = [line for line in text.splitlines() if not _DROP_LINE_RE.match(line)]
    visible = " ".join(visible_lines).strip()
    return len(visible) < 24 or visible.count("…") > 4


def _clean_document(text: str) -> str:
    lines = [line for line in text.splitlines() if not _DROP_LINE_RE.match(line)]
    return "\n".join(lines)


def _split_long_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    step = max(1, chunk_chars - overlap_chars)
    return [text[start : start + chunk_chars] for start in range(0, len(text), step)]


def split_document(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    """严格按语义段落组块；标题附到下一段，超长段落使用带重叠滑窗。"""
    text = _clean_document(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    heading = ""
    for paragraph in paragraphs:
        if paragraph.lstrip().startswith("#") and len(paragraph) <= 160:
            heading = paragraph
            continue
        section = f"{heading}\n\n{paragraph}" if heading else paragraph
        chunks.extend(_split_long_text(section, chunk_chars, overlap_chars))
    return [chunk for chunk in chunks if not is_low_quality_snippet(chunk)]


@dataclass(frozen=True)
class _IndexedChunk:
    path: str
    text: str
    normalized: str
    term_counts: Counter[str]
    length: int


class DocumentSearchIndex:
    """驻留内存的段落级 BM25 索引。构建后 search 不再访问磁盘。"""

    def __init__(
        self,
        docs_dir: str,
        *,
        max_files: int = 5000,
        max_chunks: int = 50000,
        chunk_chars: int = 800,
        overlap_chars: int = 120,
        per_file_limit: int = 2,
        min_query_coverage: float = 0.35,
        min_matched_tokens: int = 2,
        min_raw_term_coverage: float = 0.5,
    ) -> None:
        self.root = Path(docs_dir)
        self.max_files = max(1, max_files)
        self.max_chunks = max(1, max_chunks)
        self.chunk_chars = max(100, chunk_chars)
        self.overlap_chars = max(0, min(overlap_chars, self.chunk_chars // 2))
        self.per_file_limit = max(1, per_file_limit)
        self.min_query_coverage = max(0.0, min(1.0, min_query_coverage))
        self.min_matched_tokens = max(1, min_matched_tokens)
        self.min_raw_term_coverage = max(0.0, min(1.0, min_raw_term_coverage))
        self.files_indexed = 0
        self.truncated = False
        self._chunks: list[_IndexedChunk] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_length = 1.0
        self._build()

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _build(self) -> None:
        if not self.root.is_dir():
            return
        paths = sorted(
            {
                path
                for pattern in ("*.md", "*.txt")
                for path in self.root.rglob(pattern)
            },
            key=lambda path: str(path.relative_to(self.root)).lower(),
        )
        if len(paths) > self.max_files:
            paths = paths[: self.max_files]
            self.truncated = True

        total_length = 0
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            self.files_indexed += 1
            rel = str(path.relative_to(self.root))
            for chunk_text in split_document(
                text,
                chunk_chars=self.chunk_chars,
                overlap_chars=self.overlap_chars,
            ):
                tokens = tokenize(chunk_text)
                if not tokens:
                    continue
                counts = Counter(tokens)
                chunk = _IndexedChunk(
                    path=rel,
                    text=chunk_text,
                    normalized=normalize_text(chunk_text),
                    term_counts=counts,
                    length=len(tokens),
                )
                self._chunks.append(chunk)
                self._doc_freq.update(counts.keys())
                total_length += chunk.length
                if len(self._chunks) >= self.max_chunks:
                    self.truncated = True
                    break
            if len(self._chunks) >= self.max_chunks:
                break
        if self._chunks:
            self._avg_length = total_length / len(self._chunks)

    def _bm25(self, chunk: _IndexedChunk, query_tokens: set[str]) -> tuple[float, int]:
        score = 0.0
        matched = 0
        n_docs = len(self._chunks)
        k1, b = 1.5, 0.75
        for token in query_tokens:
            tf = chunk.term_counts.get(token, 0)
            if not tf:
                continue
            matched += 1
            df = self._doc_freq[token]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * chunk.length / self._avg_length)
            score += idf * tf * (k1 + 1.0) / denom
        return score, matched

    @staticmethod
    def _snippet(chunk: str, query: str, snippet_chars: int) -> str:
        compact = _SPACE_RE.sub(" ", chunk).strip()
        if len(compact) <= snippet_chars:
            return compact
        lower = compact.lower()
        candidates = [query.strip()] + sorted(
            [part for part in re.split(r"[\s,，、/；;]+", query) if len(part) >= 2],
            key=len,
            reverse=True,
        )
        pos = next(
            (lower.find(part.lower()) for part in candidates if lower.find(part.lower()) >= 0),
            0,
        )
        start = max(0, pos - snippet_chars // 3)
        end = min(len(compact), start + snippet_chars)
        start = max(0, end - snippet_chars)
        return ("..." if start else "") + compact[start:end] + ("..." if end < len(compact) else "")

    def search(self, query: str, *, top_k: int = 3, snippet_chars: int = 400) -> list[tuple[str, str]]:
        query = query.strip()
        query_tokens = set(tokenize(query))
        if not query_tokens or not self._chunks:
            return []
        query_norm = normalize_text(query)
        raw_terms = {
            term.lower()
            for term in re.split(r"[\s,，、/；;]+", query)
            if len(term.strip()) >= 2
        }
        ranked: list[tuple[float, float, str, int, _IndexedChunk]] = []
        for index, chunk in enumerate(self._chunks):
            bm25, matched = self._bm25(chunk, query_tokens)
            if matched == 0:
                continue
            coverage = matched / len(query_tokens)
            phrase_boost = 6.0 if len(query_norm) >= 4 and query_norm in chunk.normalized else 0.0
            term_coverage = (
                sum(term in chunk.text.lower() for term in raw_terms) / len(raw_terms)
                if raw_terms
                else 0.0
            )
            # 单一专业词（如 ILD/MRB）允许精确 token 命中；多 token query 则必须满足
            # 短语命中，或同时达到 token/原始词覆盖门槛，避免只因 Sample 等常见词返回噪声。
            required_matches = min(self.min_matched_tokens, len(query_tokens))
            relevant = (
                len(query_tokens) == 1
                or phrase_boost > 0
                or (
                    matched >= required_matches
                    and (
                        coverage >= self.min_query_coverage
                        or term_coverage >= self.min_raw_term_coverage
                    )
                )
            )
            if not relevant:
                continue
            filename_boost = sum(term in chunk.path.lower() for term in raw_terms) * 0.5
            score = bm25 * (0.5 + coverage) + phrase_boost + 2.0 * term_coverage + filename_boost
            ranked.append((-score, -coverage, chunk.path, index, chunk))
        ranked.sort()

        results: list[tuple[str, str]] = []
        per_file: Counter[str] = Counter()
        seen: set[tuple[str, str]] = set()
        for _, _, _, _, chunk in ranked:
            if per_file[chunk.path] >= self.per_file_limit:
                continue
            snippet = self._snippet(chunk.text, query, max(80, snippet_chars))
            key = (chunk.path, normalize_text(snippet[:160]))
            if key in seen:
                continue
            seen.add(key)
            per_file[chunk.path] += 1
            results.append((chunk.path, snippet))
            if len(results) >= top_k:
                break
        return results
