"""段落级文档索引的相关性、切块和驻留内存行为。"""

from __future__ import annotations

from pathlib import Path

from common.doc_search import DocumentSearchIndex, split_document

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "doc_search"


def test_bm25_prefers_chunk_covering_multiple_query_terms():
    index = DocumentSearchIndex(str(FIXTURE_DIR), chunk_chars=200)
    results = index.search("控制图 永久排除 Sample", top_k=2)
    assert results
    assert results[0][0] == "control.md"
    assert "永久排除" in results[0][1]


def test_index_search_does_not_reread_files(monkeypatch):
    index = DocumentSearchIndex(str(FIXTURE_DIR))

    def fail_read(*args, **kwargs):
        raise AssertionError("search 阶段不应再次读取文档")

    monkeypatch.setattr(Path, "read_text", fail_read)
    results = index.search("MRB wafer 数量")
    assert results and "两种分类" in results[0][1]


def test_long_paragraph_is_split_with_bounded_chunks():
    chunks = split_document("控制图" * 300, chunk_chars=200, overlap_chars=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_single_common_term_does_not_satisfy_multi_term_query():
    index = DocumentSearchIndex(str(FIXTURE_DIR), chunk_chars=200)
    results = index.search("Sample 完全不存在术语")
    assert results == []


def test_single_specialized_token_remains_searchable():
    index = DocumentSearchIndex(str(FIXTURE_DIR))
    results = index.search("MRB")
    assert results and results[0][0] == "mrb.md"
