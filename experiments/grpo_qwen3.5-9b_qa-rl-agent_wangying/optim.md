# GRPO QA Agent 实验优化记录

## 总体思路

本实验从**单轮 QA baseline**（`grpo_qwen3.5-9b_qa-rl_v1`）升级为**多轮检索 Agent**。优化目前推进四轮：

| 轮次 | 核心目标 | 注释 |
|------|----------|--------|
| 第一轮 | 搭好多轮 Agent 骨架 | 让模型会 search、会 boxed、能训练 |
| 第二轮 | 约束 Agent 行为协议 | 让模型先搜再答、防止空转、少乱搜|
| 第三轮 | 增强检索质量与防幻觉 | 让模型搜得准、信真实结果 |
| 第四轮 | MRB case 补丁 | 避免模型抄占位符、0 命中不瞎猜 |


## 1. 多轮 Agent 基础实现

### 实验思路

单轮 baseline 无法利用 `/data/docs`。参考 `agent-grpo_qwen3.5-9b_sliding-puzzle_v1` 的多轮 GRPO 结构，实现 `QAAgentEnv`，建立检索、阅读、作答闭环。

### 关键改动

- `common/environments/qa_agent_env.py` — 多轮 Ray 环境
- `experiments/grpo_qwen3.5-9b_qa-rl-agent_wangying/run.py` — Dataset + GRPO 入口
- `experiments/grpo_qwen3.5-9b_qa-rl-agent_wangying/config.yaml` — LoRA GRPO 配置
- `experiments/grpo_qwen3.5-9b_qa-rl-agent_wangying/run.sh` — 集群入口

- `<search>关键词</search>` → 环境 grep `/data/docs` 下 `.md` → 回灌 `[检索结果]`
- 最终 `\boxed{答案}` → `common/rewards/` 判分（含 LLM judge）
- `STOP_STRINGS = ["</search>"]`，每轮在 `</search>` 截断
- `max_searches: 3`，`search_top_k: 3`，`snippet_chars: 400`

**Prompt**

- 规定 search → 阅读结果 → `\boxed{}` 三步流程

### 训练结果

| 指标 | 数值 |
|------|------|
| step 0 accuracy | 47.3% |
| **峰值 accuracy** | **64.0%** |

### 验证case观察

- 多轮链路跑通，但行为不稳定：不检索直接答、重复 search、只 search 不 boxed。

---

## 2. Agent 行为协议约束

### 实验思路

**把 Agent 行为协议训对**。验证case包含三类失败：不检索直接答、ILD 式空转 search、无限 search 不 boxed。

### 关键改动

**`qa_agent_env.py`**

| 功能 | 实现 | 作用 |
|------|------|------|
| 必须先 search 再答 | `require_search_before_answer: true` | 未检索就 `\boxed{}` → 惩罚 |
| 重复 search 拦截 | `last_search_query` | 相同关键词不消耗次数 |
| 检索词过短拦截 | `min_search_len: 4` | 拒绝 `ILD` 等缩写 |
| 超限强制结束 | `max_searches` | 超限仍 search → 终止 episode |
| 题干将检索建议 | `_suggest_search_from_query()` | 无效动作时提示换词 |

**`run.py` Prompt**

- 至少 1 次、最多 3 次 search；禁止不检索就 boxed、重复 search、只有 search 没有 boxed

**`config.yaml`**

```yaml
env.qa_agent.cfg:
  require_search_before_answer: true
  invalid_action_penalty: -0.1
  min_search_len: 4
grpo:
  max_num_steps: 400
```

### 训练结果

| 指标 | 数值 |
|------|------|
| step 0 accuracy | 10.5%（强制 search-first，下降属预期） |
| **峰值 accuracy** | **67.7**（step 100） |
| step 400 accuracy | ~63–64%（后期漂移） |

**分析**：行为协议是 accuracy 从低位拉到 ~68% 的关键一步。

---

## 3. 检索增强 + 防幻觉

### 实验思路

分析验证失败样本，归纳三类根因：检索词不对、检索失败后幻觉作答、search 带入选项名或搜到目录/OCR 噪声。优化prompt/检索 fallback/拆词/片段过滤/训练参数。

### 关键改动

**`run.py` Prompt**

- 禁止伪造 `[检索结果]`；每次只搜 10~30 字短主题；不带选项名进 search
- 答案只能来自环境返回的真实 `[检索结果]`

**`qa_agent_env.py`**

| 功能 | 实现 |
|------|------|
| 0 结果自动 fallback | `_run_doc_search()` + `auto_fallback_search` |
| 长 query 拆词 | `_split_search_queries()` |
| 过滤低质量片段 | `_is_low_quality_snippet()`（目录/OCR 噪声） |
| 选项名带偏检测 | `_search_biased_by_options()` |
| 扩大检索 | `.md` + `.txt`，`search_top_k: 5`，`snippet_chars: 600` |

**`config.yaml`**

```yaml
grpo:
  max_num_steps: 250
  max_val_samples: 128
env.qa_agent.cfg:
  search_top_k: 5
  snippet_chars: 600
  auto_fallback_search: true
  split_long_search: true
```

### 训练结果

| 指标 | 数值 |
|------|------|
| step 0 accuracy | -9.2% |
| **峰值 accuracy** | **68.5%** |
| 对比第二轮 | 峰值持平，失败 case（安全卫生/MRB/控制图）有针对性改进 |

---

## 4. MRB 填空 case 补丁

### 实验思路

第三轮验证中 MRB 填空仍 0 分：模型把 prompt 占位符「题干关键词」原样 search、thinking 里伪造检索结果、0 命中后常识瞎答。针对 **prompt 误导 + 建议 query 过长 + 0 命中未拦截** 做环境补丁。

### 关键改动

**`run.py` Prompt**

- 示例改为 `<search>MRB wafer 数量 分类</search>`；禁止把说明文字写进 search
- 仍有检索次数且从未命中时，不得凭常识作答

**`qa_agent_env.py`**

| 功能 | 实现 |
|------|------|
| 占位符 search 拦截 | `_is_placeholder_search()`（不消耗检索次数） |
| 短检索词建议 | `_compact_search_suggest()`（去 `【1】【2】` 占位符） |
| 0 命中禁止 boxed | `has_search_hit` + `require_search_hit_before_answer` |

**`config.yaml`**

```yaml
env.qa_agent.cfg:
  require_search_hit_before_answer: true
grpo:
  max_num_steps: 400 
```

### 训练结果

| 指标 | 数值 |
|------|------|
| 训练步数 | 400 |
| **峰值 accuracy** | **67.6%** |

**分析**：补丁逻辑合理，但全量重训 + 更严约束使 step 0 与早期 rollout 更难，后续训练步数需要继续优化

---

## 主要失败模式与对策

| 失败模式 | 典型题型 | 已实施对策 | 仍可能失分原因 |
|----------|----------|------------|----------------|
| 不检索直接答 | 单选 | 必须先 search | 偶发仍 boxed |
| search 空转 | ILD 填空 | 过短/重复拦截 | 换词不当 |
| 检索 0 命中 | 多选/填空 | fallback + 短词建议 | grep 与文档表述不一致 |
| 幻觉 `[检索结果]` | MRB/控制图 | Prompt + 第四轮拦截 | thinking 内仍可能编造 |
| search 带选项名 | 控制图单选 | 选项名检测 | 间接关键词仍带偏 |
| 片段为目录/OCR | 控制图 | 低质量过滤 | 正文片段仍不够贴题 |

---

## 涉及文件

| 文件 | 作用 |
|------|------|
| `common/environments/qa_agent_env.py` | 多轮环境 + 检索逻辑（四轮迭代） |
| `experiments/.../run.py` | Dataset、System Prompt、GRPO 入口 |
| `experiments/.../config.yaml` | 训练与环境超参 |
| `experiments/.../run.sh` | 集群提交入口 |

---

## 持续优化方向

当前瓶颈已从「会不会用工具」转为「检索能否命中 + 答案能否与文档对齐」。后续将从以下方向继续：

### 1. 检索层升级

| 方向 | 说明 |
|------|------|
| **BM25 / 向量检索** | 替代简单 grep，缓解「关键词与文档表述不一致」导致的 0 命中 |
| **同义词 / 缩写扩展** | 如 ILD、MRB、FOUP 等缩写自动展开后再搜 |
| **题型感知检索** | 填空题抽名词、多选题抽主题词，而非整句题干 |
| **片段重排序** | 对 top_k 结果用 cross-encoder 或 LLM 重排，减少目录/OCR 误命中 |

### 2. 训练与采样策略

| 方向 | 说明 |
|------|------|
| **课程学习** | 先训单选/判断，再混入多选/填空/简答 |
| **reward shaping** | 对「有效 search 且命中」给中间奖励，而不只在最终 boxed 判分 |
| **验证集驱动迭代** | 每轮只针对 top 失败题型改环境/Prompt，小步 submit 对比 |

### 3. Prompt 与环境协议

| 方向 | 说明 |
|------|------|
| **检索结果结构化** | 环境回灌 JSON 或固定字段，降低模型「读错片段」概率 |
| **thinking 与作答分离** | 约束 thinking 不得出现 `[检索结果]` 字样（需推理模板配合） |

