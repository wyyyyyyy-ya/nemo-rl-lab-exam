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

from common.doc_search import DocumentSearchIndex, is_low_quality_snippet
from common.qa_agent_state import (
    credited_search_hit as _credited_search_hit,
    fallback_eligible as _fallback_eligible,
    last_assistant_text as _last_assistant_text,
    next_action_stop_strings as _next_action_stop_strings,
)
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
    has_search_hit: bool
    fallback_count: int


_FILL_BLANK_PLACEHOLDER = re.compile(r"【\d+】")
_SUGGEST_SKIP_TOKENS = frozenset(
    {"根据", "下面", "一道", "填空题", "选择题", "单选题", "多选题", "两种", "三种", "四种", "题目"}
)
_PLACEHOLDER_SEARCH_NORMALIZED = frozenset(
    {
        "题干关键词",
        "关键词",
        "专业名词",
        "从题目提取的专业名词",
        "题目原文或专业名词",
        "题干",
        "search",
        "检索",
    }
)


def _compact_search_suggest(text: str, max_len: int = 30) -> str:
    """把题面压成适合 grep 的短检索词（去填空占位符、去套话）。"""
    text = _FILL_BLANK_PLACEHOLDER.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    tokens = re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}", text)
    buf = ""
    for t in tokens:
        if t in _SUGGEST_SKIP_TOKENS:
            continue
        candidate = f"{buf} {t}".strip() if buf else t
        if len(candidate) > max_len:
            break
        buf = candidate
    return buf or text[:max_len]


def _suggest_search_from_query(query: str) -> str:
    """从题面抽取更具体的检索建议。"""
    m = re.search(r"题目[：:](.*?)(?:\n\n选项|\n选项：|\n选项:|\Z)", query, re.DOTALL)
    if m:
        return _compact_search_suggest(m.group(1).strip())
    m2 = re.search(r"题目[：:](.+)", query)
    if m2:
        return _compact_search_suggest(m2.group(1).strip())
    return _compact_search_suggest(query.strip())


def _is_placeholder_search(query: str) -> bool:
    """检测模型是否把 prompt 占位符原样当作 search 内容。"""
    return _normalize_search_query(query) in _PLACEHOLDER_SEARCH_NORMALIZED


def _normalize_search_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _is_short_search(query: str, min_len: int) -> bool:
    compact = re.sub(r"\s+", "", query)
    return len(compact) < min_len


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


def _is_low_quality_snippet(snippet: str) -> bool:
    return is_low_quality_snippet(snippet)


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


def search_markdown_docs(
    docs_dir: str,
    keyword: str,
    *,
    top_k: int = 3,
    snippet_chars: int = 400,
    max_files: int = 500,
) -> list[tuple[str, str]]:
    """兼容旧调用的便捷入口；生产环境使用 Actor 内驻留的 DocumentSearchIndex。"""
    index = DocumentSearchIndex(docs_dir, max_files=max_files)
    return index.search(keyword, top_k=top_k, snippet_chars=snippet_chars)


def _run_doc_search(
    doc_index: DocumentSearchIndex,
    keyword: str,
    *,
    top_k: int,
    snippet_chars: int,
    split_long_search: bool,
    split_max_len: int,
    allow_fallback: bool,
    fallback_query: str,
) -> tuple[list[tuple[str, str]], str, bool]:
    """执行自主检索；仅当调用方允许且自主检索无结果时额外执行一次 fallback。"""
    queries = (
        _split_search_queries(keyword, max_len=split_max_len)
        if split_long_search and len(keyword.strip()) > split_max_len
        else [keyword]
    )
    merged: list[tuple[str, str]] = []
    for q in queries:
        merged = _merge_search_results(
            merged,
            doc_index.search(q, top_k=top_k, snippet_chars=snippet_chars),
            top_k=top_k,
        )
        if len(merged) >= top_k:
            break

    effective = keyword
    used_fallback = False
    if not merged and allow_fallback:
        fb = fallback_query.strip()
        if fb and _normalize_search_query(fb) != _normalize_search_query(keyword):
            effective = fb
            used_fallback = True
            merged = doc_index.search(fb, top_k=top_k, snippet_chars=snippet_chars)
    return merged, effective, used_fallback


def format_search_results(
    keyword: str,
    results: list[tuple[str, str]],
    *,
    used_query: str = "",
    used_fallback: bool = False,
    fallback_counts_as_hit: bool = False,
    searches_remaining: int = 0,
) -> str:
    if not results:
        lines = [
            "[检索结果]",
            f"未找到与「{keyword}」相关的资料。",
        ]
        if used_fallback:
            lines.append(
                f"三次自主检索均未找到结果，环境已额外使用「{used_query}」兜底检索，仍未命中。"
            )
            lines.append("自主检索额度已耗尽，请根据已有信息输出 \\boxed{答案}。")
        elif searches_remaining > 0:
            lines.append(
                f"还可自主检索 {searches_remaining} 次；请自行判断是换词继续 search，还是根据已有信息作答。"
            )
        else:
            lines.append("自主检索额度已耗尽，请根据已有信息输出 \\boxed{答案}。")
        return "\n".join(lines)
    lines = ["[检索结果]"]
    if used_fallback and used_query:
        lines.append(
            f"（三次自主检索均无结果；环境额外使用 fallback 检索词：「{used_query}」）"
        )
        if not fallback_counts_as_hit:
            lines.append(
                "该结果不计为自主检索命中，且不占三次自主检索额度；请判断片段是否足够并最终作答。"
            )
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
        index = DocumentSearchIndex(str(doc_root))
        fb_results, fb_q, used_fallback = _run_doc_search(
            index,
            "不存在的关键词xyz",
            top_k=2,
            snippet_chars=200,
            split_long_search=False,
            split_max_len=30,
            allow_fallback=True,
            fallback_query="Exclude Sample",
        )
        assert fb_results and fb_q == "Exclude Sample" and used_fallback
        q_mrb = "下面是一道填空题。\n\n题目：根据受影响wafer数量，MRB有【1】【2】两种分类"
        mrb_suggest = _suggest_search_from_query(q_mrb)
        assert "【" not in mrb_suggest
        assert "MRB" in mrb_suggest or "wafer" in mrb_suggest
        assert _is_placeholder_search("题干关键词")
        assert not _is_placeholder_search("MRB wafer 数量")
        # 历史轮次里即使出现过 boxed，当前动作仍必须只由最后一条 assistant 消息决定。
        history = [
            {"role": "assistant", "content": r"我先猜 \\boxed{A}"},
            {"role": "environment", "content": "请先检索"},
            {"role": "assistant", "content": "<search>控制图永久排除 Sample</search>"},
        ]
        current = _last_assistant_text(history)
        assert extract_boxed(current) is None
        assert _parse_search_query(current) == "控制图永久排除 Sample"
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
        self.doc_index = DocumentSearchIndex(
            self.docs_dir,
            max_files=int(self.cfg.get("search_max_files", 5000)),
            max_chunks=int(self.cfg.get("search_max_chunks", 50000)),
            chunk_chars=int(self.cfg.get("search_chunk_chars", 800)),
            overlap_chars=int(self.cfg.get("search_chunk_overlap", 120)),
            per_file_limit=int(self.cfg.get("search_per_file_limit", 2)),
            min_query_coverage=float(self.cfg.get("search_min_query_coverage", 0.35)),
            min_matched_tokens=int(self.cfg.get("search_min_matched_tokens", 2)),
            min_raw_term_coverage=float(
                self.cfg.get("search_min_raw_term_coverage", 0.5)
            ),
        )
        print(
            "[qa_agent] 文档索引: "
            f"files={self.doc_index.files_indexed} chunks={self.doc_index.chunk_count} "
            f"truncated={self.doc_index.truncated}"
        )
        self.invalid_action_penalty = float(self.cfg.get("invalid_action_penalty", -0.1))
        self.auto_fallback_search = bool(self.cfg.get("auto_fallback_search", True))
        self.fallback_after_failed_searches = int(
            self.cfg.get("fallback_after_failed_searches", self.max_searches)
        )
        self.fallback_penalty = float(self.cfg.get("fallback_penalty", -0.05))
        self.fallback_counts_as_hit = bool(
            self.cfg.get("fallback_counts_as_hit", False)
        )
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
        # 只判最后一轮最终回答，避免历史 search、分析或被拒绝的旧答案污染 reward。
        completion = _last_assistant_text(message_log)
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
        has_search_hit = bool(meta.get("has_search_hit", False))
        last_text = _last_assistant_text(message_log)
        suggest = _suggest_search_from_query(query)

        # 动作解析只看本轮输出。若历史中曾有一个被环境拒绝的 boxed，后续轮次
        # 仍应能执行 search；最终评分时 _score_final 会从完整轨迹取最后一个 boxed。
        boxed = extract_boxed(last_text)
        if boxed is not None:
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
                    meta,
                    0.0,  # 暂不惩罚超过最大搜索次数；仍拦截执行并提示最终作答。
                    False,
                )

            if last_search and _normalize_search_query(search_query) == _normalize_search_query(
                last_search
            ):
                content = (
                    f"[检索结果]\n"
                    f"您已搜过「{search_query}」。请自行决定换一个不同的检索词，"
                    "或根据已有结果输出 \\boxed{答案}。"
                )
                return (
                    {"role": "environment", "content": content},
                    meta,
                    self.invalid_action_penalty,
                    False,
                )

            # 检索词质量由 Agent 自主负责：过短、占位符或包含选项原文的 query
            # 均不再由环境提示/拦截，直接交给检索器执行。System Prompt 仍保留软约束。

            next_search_count = search_count + 1
            allow_fallback = _fallback_eligible(
                enabled=self.auto_fallback_search,
                has_prior_search_hit=has_search_hit,
                search_count_after_current=next_search_count,
                fallback_after_failed_searches=self.fallback_after_failed_searches,
            )
            results, effective_query, used_fallback = _run_doc_search(
                self.doc_index,
                search_query,
                top_k=self.search_top_k,
                snippet_chars=self.snippet_chars,
                split_long_search=self.split_long_search,
                split_max_len=self.split_max_len,
                allow_fallback=allow_fallback,
                fallback_query=suggest,
            )
            content = format_search_results(
                search_query,
                results,
                used_query=effective_query,
                used_fallback=used_fallback,
                fallback_counts_as_hit=self.fallback_counts_as_hit,
                searches_remaining=max(0, self.max_searches - next_search_count),
            )
            credited_hit = _credited_search_hit(
                results_found=bool(results),
                used_fallback=used_fallback,
                fallback_counts_as_hit=self.fallback_counts_as_hit,
            )
            new_meta: QAAgentMetadata = {
                **meta,
                "search_count": next_search_count,
                "last_search_query": search_query,
                "has_search_hit": has_search_hit or credited_hit,
                "fallback_count": int(meta.get("fallback_count", 0))
                + int(used_fallback),
            }
            search_reward = self.fallback_penalty if used_fallback else 0.0
            return (
                {"role": "environment", "content": content},
                new_meta,
                search_reward,
                False,
            )

        if search_count > 0:
            content = (
                "请自行判断已有 [检索结果] 是否充分：若不充分且仍有额度，"
                "输出新的 <search>关键词</search>；若充分，输出最终 \\boxed{答案}。"
            )
        else:
            content = (
                "请先从题目提取专业名词检索，例如 "
                f"<search>{suggest}</search>。"
                "不要把「题干关键词」等说明文字原样当作 search 内容。"
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
            meta.setdefault("has_search_hit", False)
            meta.setdefault("fallback_count", 0)
            obs, new_meta, reward, terminated = self._step_one(log, meta)
            observations.append(obs)
            next_metadata.append(new_meta)
            rewards.append(reward)
            terminateds.append(terminated)
            expected_answers.append(str(meta.get("expected_answer", "")))

        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            # NeMo-RL 会用该字段覆盖下一轮 stop_strings；None 表示取消而非沿用。
            # 因此所有仍在继续的 episode 都必须显式保留 </search>。
            next_stop_strings=_next_action_stop_strings(terminateds),
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
