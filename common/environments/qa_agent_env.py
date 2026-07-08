"""多轮 QA 文档检索 Agent 环境（NeMo-RL 0.6.0）。

交互协议（考试要求）：
  1. 模型输出 <search>关键词</search> → 环境在 docs_dir 检索 markdown，回灌 [检索结果]
  2. 可多次检索（受 max_searches 限制）
  3. 模型输出 \\boxed{答案} → 调用 common/rewards 判分，episode 结束

与 QARewardEnv 的区别：多轮、支持检索工具；最终判分逻辑与单轮 QA 一致。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, TypedDict

from common.rewards.qa_reward import extract_boxed

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SEARCH_TAG = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)


class QAAgentMetadata(TypedDict, total=False):
    query: str
    expected_answer: str
    search_count: int
    last_search_query: str


def _suggest_search_from_query(query: str) -> str:
    """从题面抽取更具体的检索建议。"""
    m = re.search(r"题目[：:](.*?)(?:\n\n选项|\n选项：|\n选项:|\Z)", query, re.DOTALL)
    if m:
        topic = re.sub(r"\s+", " ", m.group(1).strip())
        return topic[:80] if topic else query.strip()[:80]
    return query.strip()[:80]


def _normalize_search_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _is_short_search(query: str, min_len: int) -> bool:
    compact = re.sub(r"\s+", "", query)
    return len(compact) < min_len


def _last_assistant_text(message_log: LLMMessageLogType) -> str:
    for msg in reversed(message_log):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def _all_assistant_text(message_log: LLMMessageLogType) -> str:
    parts = [
        str(msg.get("content", "")).strip()
        for msg in message_log
        if msg.get("role") == "assistant" and str(msg.get("content", "")).strip()
    ]
    return "\n".join(parts)


def _parse_search_query(text: str) -> str | None:
    m = _SEARCH_TAG.search(text)
    if not m:
        return None
    query = m.group(1).strip()
    return query or None


def _snippet_around(text: str, pos: int, radius: int) -> str:
    start = max(0, pos - radius // 2)
    end = min(len(text), pos + radius // 2)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _extract_option_terms(query: str) -> list[str]:
    """从题面选项行提取文本，用于检测 search 是否带入选项名称。"""
    terms: list[str] = []
    for m in re.finditer(r"^[A-L][\.、．\)]\s*(.+)$", query, re.MULTILINE):
        text = re.sub(r"\s+", " ", m.group(1).strip())
        if len(text) >= 4:
            terms.append(text)
    return terms


def _search_biased_by_options(search_query: str, query: str) -> str | None:
    sq = _normalize_search_query(search_query)
    for opt in _extract_option_terms(query):
        if _normalize_search_query(opt) in sq:
            return opt
    return None


def _split_search_queries(keyword: str, max_len: int = 30) -> list[str]:
    """长检索词拆成多个短 query。"""
    keyword = keyword.strip()
    if len(keyword) <= max_len:
        return [keyword]
    chunks = [c for c in re.split(r"[\s,，、；;]+", keyword) if c.strip()]
    parts: list[str] = []
    buf = ""
    for c in chunks:
        if len(buf) + len(c) + 1 <= max_len:
            buf = f"{buf} {c}".strip() if buf else c
        else:
            if buf:
                parts.append(buf)
            buf = c
    if buf:
        parts.append(buf)
    for c in chunks:
        if len(c) >= 4 and c not in parts:
            parts.append(c)
    return parts[:3] or [keyword[:max_len]]


_LOW_QUALITY_SNIPPET = (
    re.compile(r"\[End OCR\]", re.I),
    re.compile(r"## Page \d+", re.I),
    re.compile(r"Slide number:", re.I),
    re.compile(r"^\s*目录\s", re.I),
)


def _is_low_quality_snippet(snippet: str) -> bool:
    s = snippet.strip()
    if len(s) < 24:
        return True
    for pat in _LOW_QUALITY_SNIPPET:
        if pat.search(s):
            return True
    if s.count("…") > 4:
        return True
    return False


def _merge_search_results(
    *groups: list[tuple[str, str]],
    top_k: int,
) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    merged: list[tuple[str, str]] = []
    for group in groups:
        for rel, snippet in group:
            key = (rel, snippet[:100])
            if key in seen:
                continue
            seen.add(key)
            merged.append((rel, snippet))
            if len(merged) >= top_k:
                return merged
    return merged


def _keyword_terms(keyword: str) -> list[str]:
    terms = [t for t in re.split(r"[\s,，、/]+", keyword.strip()) if t]
    return terms or [keyword.strip()]


def _score_text(text: str, terms: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(term.lower()) for term in terms)


def search_markdown_docs(
    docs_dir: str,
    keyword: str,
    *,
    top_k: int = 3,
    snippet_chars: int = 400,
    max_files: int = 500,
    filter_low_quality: bool = True,
) -> list[tuple[str, str]]:
    """在 docs_dir 下检索 markdown/txt，返回 [(相对路径, 片段), ...]。"""
    root = Path(docs_dir)
    if not root.is_dir():
        return []

    terms = _keyword_terms(keyword)
    hits: list[tuple[int, str, str, int]] = []

    scanned = 0
    for pattern in ("*.md", "*.txt"):
        for path in root.rglob(pattern):
            if scanned >= max_files:
                break
            scanned += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            score = _score_text(text, terms)
            if score <= 0:
                continue
            rel = str(path.relative_to(root))
            pos = 0
            lower = text.lower()
            for term in terms:
                idx = lower.find(term.lower())
                if idx >= 0:
                    pos = idx
                    break
            snippet = _snippet_around(text, pos, snippet_chars)
            if filter_low_quality and _is_low_quality_snippet(snippet):
                continue
            hits.append((score, rel, snippet, pos))
        if scanned >= max_files:
            break

    hits.sort(key=lambda x: (-x[0], x[1], x[3]))
    return [(rel, snippet) for _, rel, snippet, _ in hits[:top_k]]


def _run_doc_search(
    docs_dir: str,
    keyword: str,
    *,
    top_k: int,
    snippet_chars: int,
    split_long_search: bool,
    split_max_len: int,
    auto_fallback: bool,
    fallback_query: str,
) -> tuple[list[tuple[str, str]], str]:
    """执行检索：可选拆词、0 命中时题干将 fallback。返回 (results, effective_query)。"""
    queries = (
        _split_search_queries(keyword, max_len=split_max_len)
        if split_long_search and len(keyword.strip()) > split_max_len
        else [keyword]
    )
    merged: list[tuple[str, str]] = []
    for q in queries:
        merged = _merge_search_results(
            merged,
            search_markdown_docs(
                docs_dir,
                q,
                top_k=top_k,
                snippet_chars=snippet_chars,
            ),
            top_k=top_k,
        )
        if len(merged) >= top_k:
            break

    effective = keyword
    if not merged and auto_fallback:
        fb = fallback_query.strip()
        if fb and _normalize_search_query(fb) != _normalize_search_query(keyword):
            merged = search_markdown_docs(
                docs_dir,
                fb,
                top_k=top_k,
                snippet_chars=snippet_chars,
            )
            if merged:
                effective = fb
    return merged, effective


def format_search_results(
    keyword: str,
    results: list[tuple[str, str]],
    *,
    suggest: str = "",
    used_query: str = "",
) -> str:
    if not results:
        lines = [
            "[检索结果]",
            f"未找到与「{keyword}」相关的资料，请换更具体的关键词后再检索，或直接给出 \\boxed{{答案}}。",
        ]
        if suggest and _normalize_search_query(suggest) != _normalize_search_query(keyword):
            lines.append(f"建议尝试：<search>{suggest}</search>")
        return "\n".join(lines)
    lines = ["[检索结果]"]
    if used_query and _normalize_search_query(used_query) != _normalize_search_query(keyword):
        lines.append(f"（已自动用题干将关键词「{used_query}」补充检索）")
    for i, (path, snippet) in enumerate(results, start=1):
        lines.append(f"{i}. {path}\n{snippet}")
    return "\n".join(lines)


def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        doc_root = Path(tmp)
        (doc_root / "control.md").write_text(
            "控制图分析时，Exclude 功能可以永久排除 Sample。Point Disable 只是临时禁用。",
            encoding="utf-8",
        )
        results = search_markdown_docs(str(doc_root), "Exclude Sample", top_k=2)
        print("search:", results)
        assert results and "control.md" in results[0][0]

        formatted = format_search_results("Exclude Sample", results)
        assert "[检索结果]" in formatted
        assert "control.md" in formatted
        print("format OK")
        assert extract_boxed("应选 A \\boxed{A}") == "A"
        q = "下面是一道单选题。\n\n题目：Carrier FOUP FOSB 创建\n\n选项：\nA. x"
        assert "Carrier" in _suggest_search_from_query(q)
        assert _is_short_search("ILD", 4)
        assert not _is_short_search("Carrier FOUP", 4)
        assert _split_search_queries("a b c d e f g h i j k l m n o p q r s t", 10)
        assert _is_low_quality_snippet("## Page 1\nSlide number: 3")
        assert not _is_low_quality_snippet("控制图 Exclude 功能可以永久排除 Sample")
        q_opt = "题目：控制图\n\n选项：\nA. Exclude\nB. Point Disable"
        assert _search_biased_by_options("Point Disable", q_opt) == "Point Disable"
        fb_results, fb_q = _run_doc_search(
            str(doc_root),
            "不存在的关键词xyz",
            top_k=2,
            snippet_chars=200,
            split_long_search=False,
            split_max_len=30,
            auto_fallback=True,
            fallback_query="Exclude Sample",
        )
        assert fb_results and fb_q == "Exclude Sample"
        print("qa_agent_env self-test OK")


if __name__ == "__main__":
    _self_test()
    raise SystemExit(0)


import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


@ray.remote  # pragma: no cover
class QAAgentEnv(EnvironmentInterface[QAAgentMetadata]):
    """多轮 QA 检索 + 判分环境（Ray Actor）。"""

    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.docs_dir = str(self.cfg.get("docs_dir", "/data/docs"))
        self.max_searches = int(self.cfg.get("max_searches", 3))
        self.search_top_k = int(self.cfg.get("search_top_k", 3))
        self.snippet_chars = int(self.cfg.get("snippet_chars", 400))
        self.min_search_len = int(self.cfg.get("min_search_len", 4))
        self.invalid_action_penalty = float(self.cfg.get("invalid_action_penalty", -0.1))
        self.truncation_penalty = float(self.cfg.get("truncation_penalty", 0.0))
        self.no_search_before_answer_penalty = float(
            self.cfg.get("no_search_before_answer_penalty", -0.1)
        )
        self.require_search_before_answer = bool(
            self.cfg.get("require_search_before_answer", True)
        )
        self.auto_fallback_search = bool(self.cfg.get("auto_fallback_search", True))
        self.split_long_search = bool(self.cfg.get("split_long_search", True))
        self.split_max_len = int(self.cfg.get("split_max_len", 30))
        self.use_judge = bool(self.cfg.get("use_judge", True))

        if self.use_judge:
            from common.rewards.qa_judge_reward import qa_judge_reward_fn

            self._reward_fn = qa_judge_reward_fn
        else:
            from common.rewards.qa_reward import qa_rule_reward_fn

            self._reward_fn = qa_rule_reward_fn

    def _score_final(
        self,
        message_log: LLMMessageLogType,
        query: str,
        expected: str,
    ) -> float:
        completion = _all_assistant_text(message_log)
        rewards = self._reward_fn([query], [completion], [expected])
        return float(rewards[0])

    def _step_one(
        self,
        message_log: LLMMessageLogType,
        meta: QAAgentMetadata,
    ) -> tuple[dict[str, str], QAAgentMetadata | None, float, bool]:
        query = str(meta.get("query", ""))
        expected = str(meta.get("expected_answer", ""))
        search_count = int(meta.get("search_count", 0))
        last_search = str(meta.get("last_search_query", ""))
        last_text = _last_assistant_text(message_log)
        suggest = _suggest_search_from_query(query)

        boxed = extract_boxed(_all_assistant_text(message_log))
        if boxed is not None:
            if self.require_search_before_answer and search_count == 0:
                content = (
                    "请先至少检索一次资料：<search>关键词</search>，"
                    "阅读 [检索结果] 后再给出 \\boxed{答案}。"
                    f"建议：<search>{suggest}</search>"
                )
                return (
                    {"role": "environment", "content": content},
                    meta,
                    self.no_search_before_answer_penalty,
                    False,
                )
            reward = self._score_final(message_log, query, expected)
            obs = {"role": "environment", "content": f"得分: {reward:.3f}"}
            return obs, None, reward, True

        search_query = _parse_search_query(last_text)
        if search_query is not None:
            if search_count >= self.max_searches:
                content = (
                    f"[检索结果]\n"
                    f"已达最大检索次数（{self.max_searches}），不得继续 search。"
                    f"请根据已有信息直接输出 \\boxed{{答案}}。"
                )
                return (
                    {"role": "environment", "content": content},
                    None,
                    self.truncation_penalty,
                    True,
                )

            if last_search and _normalize_search_query(search_query) == _normalize_search_query(
                last_search
            ):
                content = (
                    f"[检索结果]\n"
                    f"您已搜过「{search_query}」，请换更具体的关键词，或直接 \\boxed{{答案}}。"
                    f"建议：<search>{suggest}</search>"
                )
                return (
                    {"role": "environment", "content": content},
                    meta,
                    self.invalid_action_penalty,
                    False,
                )

            if _is_short_search(search_query, self.min_search_len):
                content = (
                    f"[检索结果]\n"
                    f"检索词「{search_query}」过短，请用更完整的专业名词或题干关键词。"
                    f"建议：<search>{suggest}</search>"
                )
                return (
                    {"role": "environment", "content": content},
                    meta,
                    self.invalid_action_penalty,
                    False,
                )

            biased_opt = _search_biased_by_options(search_query, query)
            if biased_opt:
                content = (
                    f"[检索结果]\n"
                    f"检索词包含选项名称「{biased_opt}」，请改用中性关键词（如题目主题），"
                    f"不要把选项原文写进 search。"
                    f"建议：<search>{suggest}</search>"
                )
                return (
                    {"role": "environment", "content": content},
                    meta,
                    self.invalid_action_penalty,
                    False,
                )

            results, effective_query = _run_doc_search(
                self.docs_dir,
                search_query,
                top_k=self.search_top_k,
                snippet_chars=self.snippet_chars,
                split_long_search=self.split_long_search,
                split_max_len=self.split_max_len,
                auto_fallback=self.auto_fallback_search,
                fallback_query=suggest,
            )
            content = format_search_results(
                search_query,
                results,
                suggest=suggest,
                used_query=effective_query,
            )
            new_meta: QAAgentMetadata = {
                **meta,
                "search_count": search_count + 1,
                "last_search_query": search_query,
            }
            return {"role": "environment", "content": content}, new_meta, 0.0, False

        if search_count > 0:
            content = "请根据已有 [检索结果] 分析，并输出最终 \\boxed{答案}。"
        else:
            content = (
                "请先用 <search>关键词</search> 检索资料。"
                f"建议：<search>{suggest}</search>"
            )
        return (
            {"role": "environment", "content": content},
            meta,
            self.invalid_action_penalty,
            False,
        )

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[QAAgentMetadata],
    ) -> EnvironmentReturn[QAAgentMetadata]:
        observations: list[dict[str, str]] = []
        next_metadata: list[QAAgentMetadata | None] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        expected_answers: list[str] = []

        for log, meta in zip(message_log_batch, metadata, strict=False):
            meta = dict(meta or {})
            meta.setdefault("search_count", 0)
            meta.setdefault("last_search_query", "")
            obs, new_meta, reward, terminated = self._step_one(log, meta)
            observations.append(obs)
            next_metadata.append(new_meta)
            rewards.append(reward)
            terminateds.append(terminated)
            expected_answers.append(str(meta.get("expected_answer", "")))

        n = len(message_log_batch)
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=[None] * n,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=expected_answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        ).float()
        if len(rewards) == 0:
            return batch, {}
        metrics = {
            "qa_agent_mean_reward": rewards.mean().item(),
            "qa_agent_perfect_rate": (rewards >= 1.0).float().mean().item(),
            "qa_agent_format_penalty_rate": (rewards < 0).float().mean().item(),
        }
        return batch, metrics
