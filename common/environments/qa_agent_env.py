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
) -> list[tuple[str, str]]:
    """在 docs_dir 下检索 markdown，返回 [(相对路径, 片段), ...]。"""
    root = Path(docs_dir)
    if not root.is_dir():
        return []

    terms = _keyword_terms(keyword)
    hits: list[tuple[int, str, str, int]] = []

    scanned = 0
    for path in root.rglob("*.md"):
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
        hits.append((score, rel, _snippet_around(text, pos, snippet_chars), pos))

    hits.sort(key=lambda x: (-x[0], x[1], x[3]))
    return [(rel, snippet) for _, rel, snippet, _ in hits[:top_k]]


def format_search_results(keyword: str, results: list[tuple[str, str]]) -> str:
    if not results:
        return (
            f"[检索结果]\n"
            f"未找到与「{keyword}」相关的资料，请换关键词或直接给出 \\boxed{{答案}}。"
        )
    lines = ["[检索结果]"]
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
        self.max_searches = int(self.cfg.get("max_searches", 5))
        self.search_top_k = int(self.cfg.get("search_top_k", 3))
        self.snippet_chars = int(self.cfg.get("snippet_chars", 400))
        self.invalid_action_penalty = float(self.cfg.get("invalid_action_penalty", -0.05))
        self.truncation_penalty = float(self.cfg.get("truncation_penalty", 0.0))
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
        last_text = _last_assistant_text(message_log)

        if extract_boxed(_all_assistant_text(message_log)) is not None:
            reward = self._score_final(message_log, query, expected)
            obs = {"role": "environment", "content": f"得分: {reward:.3f}"}
            return obs, None, reward, True

        search_query = _parse_search_query(last_text)
        if search_query is not None:
            if search_count >= self.max_searches:
                content = (
                    f"[检索结果]\n"
                    f"已达最大检索次数（{self.max_searches}），"
                    f"请根据已有信息给出 \\boxed{{答案}}。"
                )
                new_meta: QAAgentMetadata = {
                    **meta,
                    "search_count": search_count,
                }
                return {"role": "environment", "content": content}, new_meta, 0.0, False

            results = search_markdown_docs(
                self.docs_dir,
                search_query,
                top_k=self.search_top_k,
                snippet_chars=self.snippet_chars,
            )
            content = format_search_results(search_query, results)
            new_meta = {**meta, "search_count": search_count + 1}
            return {"role": "environment", "content": content}, new_meta, 0.0, False

        content = (
            "请使用 <search>关键词</search> 检索资料，"
            "或把最终答案写入 \\boxed{...}。"
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
