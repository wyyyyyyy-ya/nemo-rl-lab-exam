"""QA Agent 状态机回归测试（不依赖 Ray / NeMo-RL）。"""

from __future__ import annotations

from common.qa_agent_state import (
    credited_search_hit,
    fallback_eligible,
    last_assistant_text,
    next_action_stop_strings,
)


def test_last_assistant_text_ignores_historical_boxed_answer():
    history = [
        {"role": "assistant", "content": r"先猜 \\boxed{A}"},
        {"role": "environment", "content": "请先检索"},
        {"role": "assistant", "content": "<search>控制图永久排除 Sample</search>"},
    ]
    assert last_assistant_text(history) == "<search>控制图永久排除 Sample</search>"


def test_fallback_result_is_not_credited_by_default():
    assert not credited_search_hit(
        results_found=True,
        used_fallback=True,
        fallback_counts_as_hit=False,
    )
    assert credited_search_hit(
        results_found=True,
        used_fallback=False,
        fallback_counts_as_hit=False,
    )


def test_fallback_credit_can_restore_legacy_behavior():
    assert credited_search_hit(
        results_found=True,
        used_fallback=True,
        fallback_counts_as_hit=True,
    )


def test_fallback_only_eligible_after_three_failed_searches():
    assert not fallback_eligible(
        enabled=True,
        has_prior_search_hit=False,
        search_count_after_current=2,
        fallback_after_failed_searches=3,
    )
    assert fallback_eligible(
        enabled=True,
        has_prior_search_hit=False,
        search_count_after_current=3,
        fallback_after_failed_searches=3,
    )


def test_prior_hit_disables_fallback_even_on_third_search():
    assert not fallback_eligible(
        enabled=True,
        has_prior_search_hit=True,
        search_count_after_current=3,
        fallback_after_failed_searches=3,
    )


def test_continuing_episode_keeps_search_stop_string():
    assert next_action_stop_strings([False, True, False]) == [
        ["</search>"],
        None,
        ["</search>"],
    ]
