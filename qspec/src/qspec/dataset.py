from dataclasses import dataclass

from datasets import load_dataset


@dataclass(frozen=True)
class GSM8KSample:
    sample_id: str
    question: str
    reference_answer: str


def load_gsm8k_samples(
    split: str = "test",
    limit: int | None = None,
) -> list[GSM8KSample]:
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=split,
    )

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return [
        GSM8KSample(
            sample_id=f"{split}-{index}",
            question=row["question"],
            reference_answer=row["answer"],
        )
        for index, row in enumerate(dataset)
    ]