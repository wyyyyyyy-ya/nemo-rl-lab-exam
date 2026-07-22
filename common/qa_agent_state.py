"""QA Agent 状态机的纯 Python 辅助逻辑（不依赖 Ray / NeMo-RL）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SEARCH_ACTION_STOP_STRING = "</search>"


def last_assistant_text(message_log: Sequence[Mapping[str, Any]]) -> str:
    """仅返回当前轮（最后一条）assistant 文本，避免历史动作干扰状态解析。"""
    for message in reversed(message_log):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


def credited_search_hit(
    *,
    results_found: bool,
    used_fallback: bool,
    fallback_counts_as_hit: bool,
) -> bool:
    """只有模型原始 query 命中时才默认记功；fallback 可通过配置恢复旧行为。"""
    return results_found and (not used_fallback or fallback_counts_as_hit)


def fallback_eligible(
    *,
    enabled: bool,
    has_prior_search_hit: bool,
    search_count_after_current: int,
    fallback_after_failed_searches: int,
) -> bool:
    """只有自主搜索达到失败阈值且此前从未命中时，才允许环境执行一次 fallback。"""
    return (
        enabled
        and not has_prior_search_hit
        and search_count_after_current >= fallback_after_failed_searches
    )


def next_action_stop_strings(terminateds: Sequence[bool]) -> list[list[str] | None]:
    """为仍在运行的 episode 保留逐轮 search stop，结束样本不再设置。"""
    return [
        None if terminated else [SEARCH_ACTION_STOP_STRING]
        for terminated in terminateds
    ]
