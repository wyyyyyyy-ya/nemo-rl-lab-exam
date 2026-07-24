#!/usr/bin/env python
# 题库多轮 Agent GRPO 训练脚本（NeMo-RL 0.6.0）。
# 数据：datasets/qa_rl 的 train/val jsonl（每行 {"query", "expected_answer": "[type] ..."}）。
# 环境：common/environments/qa_agent_env.py 的 QAAgentEnv
#       （<search> 检索 /data/docs + 最终 \boxed{} 判分，简答可走 LLM 裁判）。
# 由本实验 run.sh 自动调用（本目录存在 run.py 时优先于 ENTRY）。
import argparse
import json
import os
import pprint
import sys
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import nemo_rl.algorithms.grpo as grpo_module
from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from common.environments.qa_agent_env import QAAgentEnv

TASK_NAME = "qa_agent"
# 每轮生成在 </search> 处截断，便于环境回灌检索结果后继续多轮。
STOP_STRINGS = ["</search>"]


def _install_fixed_refit_buffer() -> None:
    """Avoid the unsupported NVML free-memory query on the lab GB10."""
    original_refit = grpo_module.refit_policy_generation
    buffer_size_gb = int(os.environ.get("NRL_REFIT_BUFFER_SIZE_GB", "4"))

    def refit_with_fixed_buffer(
        policy, policy_generation, colocated_inference, *args, **kwargs
    ):
        if colocated_inference and kwargs.get("_refit_buffer_size_gb") is None:
            kwargs["_refit_buffer_size_gb"] = buffer_size_gb
        return original_refit(
            policy,
            policy_generation,
            colocated_inference,
            *args,
            **kwargs,
        )

    grpo_module.refit_policy_generation = refit_with_fixed_buffer

LONG_AGENT_SYSTEM_PROMPT = r"""
你是技术培训考题检索 Agent。目标是用尽可能少而有效的检索找到可靠依据，并按题目要求准确作答。

【可用动作】
- 检索：<search>检索词</search>
- 最终作答：\boxed{答案}
- 每一轮只能选择一种动作。不要在同一轮同时输出 search 和 boxed。

【决策流程】
1. 首次检索
   - 每题必须先自主检索至少 1 次，最多自主检索 3 次。
   - 先识别题目真正询问的对象、属性或关系，再生成检索词。
   - 检索词优先采用“核心专业概念 + 所问属性/关系”，例如：
     <search>MRB wafer 数量 分类</search>
   - 使用 4~30 字的具体短语；保留必要的中英文术语和缩写。
   - 不要照抄整道题，不要把“题干关键词”“专业名词”等说明文字当作检索词，
     也不要只用某个答案选项作为带倾向性的检索词。

2. 评估环境返回的 [检索结果]
   收到结果后，做简短的证据判断：
   - 相关性：片段是否在讨论题目中的同一对象和同一问题？
   - 充分性：片段是否直接给出作答所需的事实、条件、分类、数值或因果关系？
   - 一致性：多个片段是否互相支持；是否存在冲突、否定词或适用条件差异？
   - 缺口：若仍不能作答，当前还缺哪个概念、属性或关系？

3. 自主决定下一步
   - 证据充分：停止检索，进入最终作答。
   - 证据不足且仍有额度：围绕“缺失信息”改写检索词；新 query 必须比上一次更具体或换一个角度，禁止原样重复。
   - 结果很多但不相关：去掉宽泛词，保留题目主题和所问关系。
   - 结果为零：换用同义表达、完整术语、中文/英文名称或“概念 + 属性”重新检索。
   - 不要为了凑满三次而继续搜索；结果足够时立即作答。

4. fallback 规则
   - 只有三次自主检索都返回零结果时，环境才会额外执行一次题干 fallback；它不占三次自主额度。
   - fallback 返回后不得继续 search，应判断片段是否有用并完成最终作答。

5. 最终作答
   - 严格遵守题目要求的答案格式，只输出一次 \boxed{答案}。
   - 单选/判断只填一个字母；多选按题目要求列出字母；填空按空位顺序填写；简答覆盖关键要点。
   - 优先依据环境真实返回的 [检索结果]；若三次自主检索和 fallback 都没有可用证据，
     基于题目与已有信息给出最稳妥答案，但不得声称检索结果支持了它。
   - 禁止伪造、改写或自行输出“[检索结果]”；禁止在没有最终 boxed 的情况下结束。
""".strip()

DEFAULT_AGENT_SYSTEM_PROMPT = r"""
你是技术培训考题检索 Agent，用尽量少的有效检索寻找可靠依据并准确作答。
可用工具每轮只能二选一：<search>检索词</search> 或 \boxed{答案}；禁止同轮混用。

规则：
1. 每题先检索至少一次。识别所问对象与属性/关系，用 4~30 字的“核心概念 + 所问属性”检索，保留必要的中英文术语；不要照抄整题、使用空泛词或仅用某个选项诱导检索。
2. 对 [检索结果] 判断相关性、充分性、一致性和信息缺口。证据充分则立即作答；不足且有额度时围绕缺口换更具体的词或角度。结果过宽就收窄，结果为零就换同义词、完整术语或中英文名称；禁止重复查询或为耗尽额度而搜索。
3. 若环境在连续无结果后执行题干 fallback，收到后不得继续检索，应判断证据并完成作答。
4. 严格遵守题目要求的答案格式，最终只输出一次 \boxed{答案}：单选/判断填一个字母；多选列出所需字母；填空按空位顺序；简答覆盖关键要点。优先依据真实结果；无证据时可给出最稳妥答案，但不得假称有检索依据。
5. 禁止伪造、改写或自行输出 [检索结果]，也不得无 boxed 结束。
""".strip()

def parse_args():
    parser = argparse.ArgumentParser(description="题库多轮 Agent GRPO 训练")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class QAAgentJsonlDataset(Dataset):
    """读题库 jsonl，转成多轮 Agent 用的 DatumSpec。"""

    def __init__(
        self,
        path: str,
        tokenizer,
        input_key: str,
        output_key: str,
        system_prompt: str | None = None,
    ):
        self.rows = _read_jsonl(path)
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatumSpec:
        row = self.rows[idx]
        query = str(row[self.input_key])
        expected = str(row[self.output_key])

        chat: list[dict[str, str]] = []
        if self.system_prompt:
            chat.append({"role": "system", "content": self.system_prompt})
        chat.append({"role": "user", "content": query})

        prompt_text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=False
        ).strip()
        token_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        message_log: LLMMessageLogType = [
            {"role": "user", "content": prompt_text, "token_ids": token_ids}
        ]
        return {
            "message_log": message_log,
            "length": len(token_ids),
            "extra_env_info": {
                "expected_answer": expected,
                "query": query,
                "search_count": 0,
                "last_search_query": "",
                "has_search_hit": False,
                "fallback_count": 0,
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": STOP_STRINGS,
        }


def main():
    register_omegaconf_resolvers()
    _install_fixed_refit_buffer()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    print(f"已加载配置: {args.config}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config: MasterConfig = MasterConfig(**config)
    print("最终配置：")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"日志目录: {config.logger['log_dir']}")

    init_ray()
    set_seed(config.grpo["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit(
            "未指定数据目录。集群提交时平台会注入 QA_RL_DATA_DIR；"
            "本地调试请设置 config.data.data_dir 或 export QA_RL_DATA_DIR。"
        )
    input_key = data_cfg.get("input_key", "query")
    output_key = data_cfg.get("output_key", "expected_answer")
    system_prompt = data_cfg.get("system_prompt")
    if not system_prompt:
        system_prompt = DEFAULT_AGENT_SYSTEM_PROMPT

    train_dataset = QAAgentJsonlDataset(
        os.path.join(data_dir, "train.jsonl"),
        tokenizer,
        input_key,
        output_key,
        system_prompt,
    )
    val_dataset = QAAgentJsonlDataset(
        os.path.join(data_dir, "val.jsonl"),
        tokenizer,
        input_key,
        output_key,
        system_prompt,
    )
    print(f"训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条")

    env_cfg = config.env[TASK_NAME]["cfg"]
    env = QAAgentEnv.options(num_gpus=0).remote(cfg=dict(env_cfg))
    task_to_env = {TASK_NAME: env}

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
