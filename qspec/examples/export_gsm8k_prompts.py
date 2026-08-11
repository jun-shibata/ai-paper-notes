import json
from pathlib import Path

from qspec.dataset import load_gsm8k_samples
from qspec.prompt import build_gsm8k_prompt


def main() -> None:
    samples = load_gsm8k_samples(
        split="test",
        limit=10,
    )

    output_path = Path(
        "data/processed/gsm8k-test-10.jsonl"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for sample in samples:
            record = {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "reference_answer": (
                    sample.reference_answer
                ),
                "prompt": build_gsm8k_prompt(
                    sample.question
                ),
            }

            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")

    print(
        f"Saved {len(samples)} prompts "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()