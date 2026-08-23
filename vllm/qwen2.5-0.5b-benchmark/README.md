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


## Concurrent Request Benchmark

The OpenAI-compatible vLLM server was benchmarked with concurrency levels from 1 to 32. Each request used an input length of 64 tokens and generated 32 output tokens.

| Concurrency | Median TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
| :-: | :--- | :--- | :--- |
| 1 | 255.92 | 42.90 | 42.90 |
| 2 | 439.63 | 50.61 | 50.61 |
| 4 | 1,028.64 | 59.72 | 59.72 |
| 8 | 2,066.21 | 93.13 | 93.13 |
| 16 | 4,070.87 | 156.35 | 156.35 |
| 32 | 8,452.69 | 267.60 | 267.60 |


TTFT increased almost proportionally to concurrency from concurrency 4 onward. TPOT increased moderately up to concurrency 4, then degraded more rapidly at concurrency 8 and above.

These results are consistent with the offline batch benchmark, where throughput began to saturate around batch size 8. On this M1 CPU environment, concurrency 4 provides a reasonable balance between batching efficiency and request latency. Concurrency 8 may be appropriate when throughput is more important than interactive latency, while concurrency levels of 16 or 32 introduce substantial latency.

The unusually high P99 ITL observed at concurrency 2 appears to be a transient outlier and should be verified with repeated benchmark runs.
