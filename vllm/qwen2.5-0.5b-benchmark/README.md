# Qwen2.5 0.5B vLLM benchmark

## Environment

- Machine: Mac M1
- OS: MacOS
- Memory: 16GB
- Python: v3.12
- vLLM version: v0.27.0
- vLLM commit: 4bdc8a788d2e2ce9165d552b3d4d8b72604626bf
- PyTorch version: 2.13.0
- Model: Qwen/Qwen2.5-0.5B-Instruct
- dtype: float16
- max_model_len: 512

## Experiments

Measure throughput, elapsed time, and memory usage while varying:

- Batch size
- Prompt length
- Output length