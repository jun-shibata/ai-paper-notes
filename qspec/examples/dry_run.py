from pathlib import Path

import torch

from qspec.compare_top1 import calculate_top1_agreement
from qspec.dataset import load_gsm8k_samples
from qspec.prompt import build_gsm8k_prompt
from qspec.result import SampleResult, append_result_jsonl


def main() -> None:
    samples = load_gsm8k_samples(limit=1)

    for sample in samples:
        prompt = build_gsm8k_prompt(
            question=sample.question,
            examples=[],
        )

        logits_w4a16 = torch.tensor([
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
        ])

        logits_w4a4 = torch.tensor([
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ])

        agreement = calculate_top1_agreement(
            logits_w4a16,
            logits_w4a4,
        )

        result = SampleResult(
            sample_id=sample.sample_id,
            question=sample.question,
            generated_answer="<dry-run>",
            prompt_length=len(prompt),
            answer_length=agreement.num_tokens,
            num_matches=agreement.num_matches,
            agreement=agreement.agreement,
            top1_w4a16=agreement.top1_w4a16,
            top1_w4a4=agreement.top1_w4a4,
            matches=agreement.matches,
        )

        append_result_jsonl(
            result,
            Path("results/dry-run.jsonl"),
        )


if __name__ == "__main__":
    main()