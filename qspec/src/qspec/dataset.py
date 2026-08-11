from collections.abc import Mapping
from dataclasses import dataclass

from datasets import load_dataset


@dataclass(frozen=True)
class GSM8KSample:
    sample_id: str
    question: str
    reference_answer: str


def convert_gsm8k_row(
    row: Mapping[str, str],
    *,
    split: str,
    index: int,
) -> GSM8KSample:
    return GSM8KSample(
        sample_id=f"{split}-{index}",
        question=row["question"],
        reference_answer=row["answer"],
    )


def load_gsm8k_samples(
    split: str = "test",
    limit: int | None = None,
) -> list[GSM8KSample]:
    """
    Load samples from the GSM8K dataset.
    Args:
        split:
            Dataset split. Usually "train" or "test".
        limit:
            Maximum number of samples to return.
            If None, return all samples.
    Return:
        GSM8K samples converted to GSM8KSample objects.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=split,
    )

    if limit is not None:
        dataset = dataset.select(
            range(min(limit, len(dataset)))
        )

    return [
        convert_gsm8k_row(
            row,
            split=split,
            index=index,
        )
        for index, row in enumerate(dataset)
    ]