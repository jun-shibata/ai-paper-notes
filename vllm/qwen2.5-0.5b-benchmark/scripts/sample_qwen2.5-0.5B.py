from multiprocessing import freeze_support
import os

os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
from vllm import LLM, SamplingParams


def main() -> None:
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    llm = LLM(
        model=model,
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=0.25,
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
    )

    outputs = llm.generate(
        ["Explain continuous batching in one sentence."],
        params,
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    freeze_support()
    main()
