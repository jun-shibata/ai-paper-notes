import csv
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

from vllm import LLM, SamplingParams


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
REPETITIONS = 3
MAX_TOKENS = 32

RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "batch_size_results.csv"
)


def make_prompts(batch_size: int, repetition: int) -> list[str]:
    base_prompt = (
        "Explain continuous batching in large language model inference "
        "in two concise sentences."
    )

    # 各リクエストを少しだけ異なる内容にして、
    # 同一prefixのキャッシュ再利用による偏りを避ける。
    return [
        f"{base_prompt}\nRequest ID: batch-{batch_size}-run-{repetition}-item-{i}"
        for i in range(batch_size)
    ]


def run_once(
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int,
    repetition: int,
) -> dict:
    prompts = make_prompts(batch_size, repetition)

    started_at = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )
    elapsed_seconds = time.perf_counter() - started_at

    input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    output_tokens = sum(
        len(completion.token_ids)
        for output in outputs
        for completion in output.outputs
    )
    total_tokens = input_tokens + output_tokens

    return {
        "batch_size": batch_size,
        "repetition": repetition,
        "elapsed_seconds": elapsed_seconds,
        "requests_per_second": batch_size / elapsed_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / elapsed_seconds,
        "total_tokens": total_tokens,
        "total_tokens_per_second": total_tokens / elapsed_seconds,
    }


def print_summary(results: list[dict]) -> None:
    print("\nBenchmark summary")
    print(
        f"{'Batch':>6} "
        f"{'Median sec':>12} "
        f"{'Requests/s':>12} "
        f"{'Output tok/s':>14} "
        f"{'Total tok/s':>13}"
    )

    for batch_size in BATCH_SIZES:
        rows = [
            row for row in results
            if row["batch_size"] == batch_size
        ]

        median_elapsed = statistics.median(
            row["elapsed_seconds"] for row in rows
        )
        median_requests_per_second = statistics.median(
            row["requests_per_second"] for row in rows
        )
        median_output_tokens_per_second = statistics.median(
            row["output_tokens_per_second"] for row in rows
        )
        median_total_tokens_per_second = statistics.median(
            row["total_tokens_per_second"] for row in rows
        )

        print(
            f"{batch_size:>6} "
            f"{median_elapsed:>12.3f} "
            f"{median_requests_per_second:>12.2f} "
            f"{median_output_tokens_per_second:>14.2f} "
            f"{median_total_tokens_per_second:>13.2f}"
        )


def save_results(results: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved raw results to: {RESULTS_PATH}")


def main() -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        ignore_eos=True,
    )

    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=0.25,
        enable_prefix_caching=False,
    )

    # モデル起動直後の初回処理を測定値から除外する。
    print("Running warm-up...")
    llm.generate(
        ["Warm up the language model inference engine."],
        sampling_params,
        use_tqdm=False,
    )

    results = []

    for batch_size in BATCH_SIZES:
        for repetition in range(1, REPETITIONS + 1):
            result = run_once(
                llm=llm,
                sampling_params=sampling_params,
                batch_size=batch_size,
                repetition=repetition,
            )
            results.append(result)

            print(
                f"batch={batch_size:>2}, "
                f"run={repetition}, "
                f"elapsed={result['elapsed_seconds']:.3f}s, "
                f"output={result['output_tokens_per_second']:.2f} tok/s"
            )

    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()