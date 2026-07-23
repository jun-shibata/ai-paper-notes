from dataclasses import dataclass


@dataclass(frozen=True)
class FewShotExample:
    question: str
    answer: str


def build_gsm8k_prompt(
    question: str,
    examples: list[FewShotExample],
) -> str:
    sections: list[str] = []

    for example in examples:
        sections.append(
            f"Question: {example.question}\n"
            f"Answer: {example.answer}"
        )

    sections.append(
        f"Question: {question}\n"
        "Answer:"
    )

    return "\n\n".join(sections)