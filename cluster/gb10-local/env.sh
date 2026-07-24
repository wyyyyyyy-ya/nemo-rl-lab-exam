# 单机 DGX Spark GB10 环境。不要继承 gb10-spark 的跨节点 RoCE 网卡配置。

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_memory_usage_threshold=0.95
export RAY_memory_monitor_refresh_ms=2000
export NCCL_DEBUG=WARN
