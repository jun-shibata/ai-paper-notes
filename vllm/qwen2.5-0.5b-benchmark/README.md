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

## Measure throughput (Batch Size)

Qwen2.5-0.5B-Instruct was executed on the CPU backend of a MacBook Air M1 (16GB). Generation throughput improved from 28.69 tok/s at a batch size of 1 to 91.42 tok/s at a batch size of 8, and further to 99.87 tok/s at a batch size of 32. Beyond a batch size of 8, performance gains slowed significantly, while the batch completion time increased almost proportionally. In this environment, a batch size of 4 to 8 offers a good balance between throughput and completion time, and the maximum generation throughput appears to plateau at around 100 tok/s. (benchmark_batch_size.py)

Benchmark summary

| Batch | Median sec | Requests/s | Output tok/s | Total tok/s |
| :-: | :--- | :--- | :--- | :--- |
| 1 | 1.116 | 0.90 | 28.69 | 51.99 |
| 2 | 1.285 | 1.56 | 49.82 | 90.30 |
| 4 | 1.640 | 2.44 | 78.05 | 141.46 |
| 8 | 2.800 | 2.86 | 91.42 | 165.71 |
| 16 | 5.238 | 3.05 | 97.74 | 181.36 |
| 32 | 10.253 | 3.12 | 99.87 | 186.28 |
