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

DEFAULT_AGENT_SYSTEM_PROMPT = (
    "你是技术培训考题助手。必须严格按以下流程作答：\n"
    "1. 先用 <search>题干关键词</search> 检索技术资料（至少 1 次，最多 3 次）；\n"
    "2. 阅读 [检索结果] 后简要分析；\n"
    "3. 最后只输出一次 \\boxed{答案}。\n"
    "禁止：不检索就直接 \\boxed{}；禁止重复相同 search；禁止只有 search 没有最终 \\boxed{}。\n"
    "检索词请用题目里的专业名词或中英文术语，不要只搜 2~3 个字母的缩写。"
)


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
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": STOP_STRINGS,
        }


def main():
    register_omegaconf_resolvers()
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
        _nemo_gym,
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
