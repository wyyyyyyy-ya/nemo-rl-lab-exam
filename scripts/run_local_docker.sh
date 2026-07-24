#!/usr/bin/env bash
# 在当前单机 DGX Spark 上用已安装的 GB10 兼容镜像运行实验。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXP_NAME="${1:-grpo_qwen3.5-9b_qa-rl-agent_wangying}"
IMAGE="${NEMO_RL_IMAGE:-nemo-rl:local}"
EXP_DIR="${REPO_ROOT}/experiments/${EXP_NAME}"

[[ -d "${EXP_DIR}" ]] || { echo "实验不存在: ${EXP_DIR}" >&2; exit 2; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
  echo "镜像不存在: ${IMAGE}" >&2
  echo "当前 GB10 请使用本机已构建的 nemo-rl:local；官方 v0.6.0 镜像的 PyTorch 不含 sm_121。" >&2
  exit 2
}
[[ -d /data/datasets/qa_rl ]] || { echo "缺少 /data/datasets/qa_rl" >&2; exit 2; }
[[ -d /data/docs ]] || { echo "缺少 /data/docs" >&2; exit 2; }

mkdir -p "${REPO_ROOT}/outputs"

ENV_ARGS=()
for name in HF_TOKEN JUDGE_BASE_URL JUDGE_MODEL JUDGE_API_KEY JUDGE_CONCURRENCY JUDGE_TIMEOUT; do
  [[ -n "${!name:-}" ]] && ENV_ARGS+=(--env "${name}")
done

exec docker run --rm --gpus all \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "${REPO_ROOT}:${REPO_ROOT}" \
  --volume /data:/data \
  --workdir "${REPO_ROOT}" \
  --env NEMO_RL_DIR=/opt/nemo-rl \
  --env CLUSTER_PROFILE=gb10-local \
  --env HF_HOME=/data/hf_cache \
  --env OUTPUT_ROOT="${REPO_ROOT}/outputs" \
  "${ENV_ARGS[@]}" \
  "${IMAGE}" \
  bash "${EXP_DIR}/run.sh"
